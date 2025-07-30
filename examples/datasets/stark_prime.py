import os
import json
import shutil
import random
import subprocess
import os.path as osp
from functools import partial
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from optimas.wrappers.example import Example


class STaRKDataset:
    def __init__(self, name: str, root: Optional[str] = None, human_generated_eval: bool = False):
        self.name = name
        self.root = root
        self.human_generated_eval = human_generated_eval
        self.dataset_root = osp.join(root, name) if root else None
        self._download()

        self.query_dir = osp.join(self.dataset_root, 'stark_qa')
        filename = 'stark_qa_human_generated_eval.csv' if human_generated_eval else 'stark_qa.csv'
        self.qa_csv_path = osp.join(self.query_dir, filename)

        self.data = pd.read_csv(self.qa_csv_path)
        self.indices = sorted(self.data['id'].tolist())
        self.split_dir = osp.join(self.dataset_root, 'split')
        self.split_indices = self.get_idx_split()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = self.data[self.data['id'] == self.indices[idx]].iloc[0]
        return row['query'], row['id'], eval(row['answer_ids']), None

    def get_idx_split(self, test_ratio: float = 1.0) -> Dict[str, torch.Tensor]:
        if self.human_generated_eval:
            return {'human_generated_eval': torch.LongTensor(self.indices)}

        split_idx = {}
        for split in ['train', 'val', 'test', 'test-0.1']:
            with open(osp.join(self.split_dir, f'{split}.index'), 'r') as f:
                ids = [int(i) for i in f.read().split()]
            split_idx[split] = torch.LongTensor([self.indices.index(i) for i in ids])

        if test_ratio < 1.0:
            split_idx['test'] = split_idx['test'][:int(len(split_idx['test']) * test_ratio)]

        return split_idx

    def _download(self):
        self.dataset_root = download_hf_folder(
            repo="snap-stanford/stark",
            folder=f"qa/{self.name}",
            repo_type="dataset",
            save_as_folder=self.dataset_root,
        )


def download_hf_folder(repo: str, folder: str, repo_type: str = "dataset", save_as_folder: Optional[str] = None) -> str:
    files = list_repo_files(repo, repo_type=repo_type)
    folder_files = [f for f in files if f.startswith(folder + '/')]

    for file in folder_files:
        path = hf_hub_download(repo, file, repo_type=repo_type)
        if save_as_folder:
            local_path = os.path.join(save_as_folder, os.path.relpath(file, folder))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not osp.exists(local_path):
                shutil.copy2(path, local_path)
        else:
            save_as_folder = osp.dirname(osp.dirname(path))

    return save_as_folder


def get_vss_topk(vss, query: str, query_id: int) -> Dict[int, float]:
    scores = vss(query, query_id)
    return dict(sorted({int(k): float(v) for k, v in scores.items()}.items(), key=lambda x: x[1], reverse=True))


def process_single_example(args):
    idx, qa_dataset, vss = args
    query, qid, *_ = qa_dataset[idx]
    return idx, get_vss_topk(vss, query, qid)


def filter_candidates(candidates: List[int], answer_ids: List[int], n_answers: int = 1, n_nonanswers: int = 4) -> List[int]:
    answers = [cid for cid in candidates if cid in answer_ids]
    nonanswers = [cid for cid in candidates if cid not in answer_ids]
    selected = random.sample(answers, n_answers) if len(answers) >= n_answers else [random.choice(answer_ids)]
    remaining = [cid for cid in candidates if cid not in selected]
    non_selected = random.sample(remaining, min(len(remaining), 5 - len(selected)))
    while len(selected) + len(non_selected) < 5:
        non_selected.append(random.choice(candidates))
    result = selected + non_selected
    random.shuffle(result)
    return result


def process_item(i, candidate_id_dict, qa_dataset, get_rel_info, get_text_info):
    candidates = list(candidate_id_dict[i].keys())[:5]
    answer_ids = qa_dataset[i][2]
    filtered = filter_candidates(candidates, answer_ids)
    return i, {
        "candidate_ids": filtered,
        "simiarity": [candidate_id_dict[i][cid] for cid in filtered],
        "relation_info": [get_rel_info(cid) for cid in filtered],
        "text_info": [get_text_info(cid) for cid in filtered],
    }


def process_split_parallel(
    indices: List[int],
    candidate_dict: Dict[int, Dict[int, float]],
    qa_dataset: STaRKDataset,
    get_rel_info,
    get_text_info,
    desc: str,
    max_workers: int = 128
) -> Dict[int, Dict[str, Any]]:
    results = {}
    func = partial(process_item, candidate_id_dict=candidate_dict, qa_dataset=qa_dataset,
                   get_rel_info=get_rel_info, get_text_info=get_text_info)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, i): i for i in indices}
        for future in tqdm(as_completed(futures), total=len(indices), desc=desc):
            idx, res = future.result()
            results[idx] = res
    return results


def truncate_rel_text_info(info_dict: Dict[int, Dict[str, List[str]]], limit: int = 1024) -> Dict:
    truncate = lambda text: text[:limit] if len(text) > limit else text
    for val in info_dict.values():
        val["text_info"] = [truncate(t) for t in val["text_info"]]
        val["relation_info"] = [truncate(t) for t in val["relation_info"]]
    return info_dict


def dataset_engine(root='examples/data/stark', **kwargs) -> Tuple[List[Example], List[Example], List[Example]]:
    seed = kwargs.get("seed", 42)
    random.seed(seed)

    # Download embeddings to `emb_dir`
    # following https://github.com/snap-stanford/stark/blob/main/emb_download.py
    # => examples/systems/stark/functional/emb_download.py
    emb_dir = osp.join(root, "emb")
    stark_data_path = osp.join(root, "processed")

    os.makedirs(stark_data_path, exist_ok=True)

    candidate_id_paths = {split: osp.join(stark_data_path, f"{split}_ids.json") for split in ["train", "val", "test"]}
    filtered_paths = {split: osp.join(stark_data_path, f"filtered_{split}_ids.json") for split in ["train", "val", "test"]}

    qa_dataset = STaRKDataset("prime")

    node_emb_dir = osp.join(emb_dir, "prime", "text-embedding-ada-002", "doc")
    query_emb_dir = osp.join(emb_dir, "prime", "text-embedding-ada-002", "query")

    idx_split = qa_dataset.get_idx_split()
    idx_split = {
        "train": [int(i) for i in idx_split["train"][:250]],
        "val": [int(i) for i in idx_split["val"][:25]],
        "test": [int(i) for i in idx_split["test"][:100]]
    }

    if not all(osp.exists(p) for p in candidate_id_paths.values()):
        from stark_qa.models import VSS
        vss = VSS(kb, query_emb_dir, node_emb_dir, emb_model=emb_model, device="cpu")
        for split in ["train", "val", "test"]:
            args = [(i, qa_dataset, vss) for i in idx_split[split]]
            candidate_dict = {}
            with ThreadPoolExecutor(max_workers=128) as executor:
                futures = [executor.submit(process_single_example, a) for a in args]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} set"):
                    i, result = fut.result()
                    candidate_dict[i] = result
            with open(candidate_id_paths[split], "w") as f:
                json.dump(candidate_dict, f)
    else:
        candidate_dict = {split: json.load(open(path)) for split, path in candidate_id_paths.items()}

    if not all(osp.exists(p) for p in filtered_paths.values()):
        
        from stark_qa import load_skb
        from examples.systems.stark.tools import GetRelationInfo, GetTextInfo
        skb = load_skb("prime")
        get_rel_info = GetRelationInfo(skb)
        get_text_info = GetTextInfo(skb)

        results = {}
        for split in ["train", "val", "test"]:
            results[split] = process_split_parallel(
                idx_split[split],
                candidate_dict[split],
                qa_dataset,
                get_rel_info,
                get_text_info,
                f"Filtering {split} candidates"
            )
            with open(filtered_paths[split], "w") as f:
                json.dump(results[split], f)
    else:
        results = {split: json.load(open(path)) for split, path in filtered_paths.items()}

    for split in results:
        results[split] = truncate_rel_text_info({int(k): v for k, v in results[split].items()}, 1024)

    def build_examples(info, split_name):
        return [
            Example(
                question=qa_dataset[i][0],
                question_id=qa_dataset[i][1],
                answer_ids=qa_dataset[i][2],
                candidate_ids=info[i]["candidate_ids"],
                relation_info=json.dumps(info[i]["relation_info"], indent=2),
                text_info=json.dumps(info[i]["text_info"], indent=2),
                emb_scores=[round(float(v), 2) for v in info[i]["simiarity"]],
            ).with_inputs("question", "relation_info", "text_info", "emb_scores")
            for i in info
        ]

    trainset = build_examples(results["train"], "train")
    valset = build_examples(results["val"], "val")
    testset = build_examples(results["test"], "test")

    print(f"[STaRK] Loaded {len(trainset)} train, {len(valset)} val, {len(testset)} test examples.")
    return trainset, valset, testset


if __name__ == "__main__":
    trainset, valset, testset = dataset_engine()
    print(f"Loaded {len(trainset)} training, {len(valset)} validation, and {len(testset)} test examples.")
    print(trainset[0])
    # => 
    # Example({'question': 'Is the lens-specific intermediate filament-like protein, filensin, which is encoded by a gene expressed in the lens of camera-type eyes but not found in nasal cavity epithelial cells, involved in providing structural support to the ocular lens?', 
    # 'question_id': 9117, 
    # 'answer_ids': [4607], 
    # 'candidate_ids': [2241, 12670, 34670, 120673, 4607], 
    # 'relation_info': '[ "- name: BFSP2- type: gene/protein- relations:  ppi: {gene/protein: (MAX, ...),} ...', 
    # 'text_info': '[ "- name: BFSP2...  - summary (protein summary text): More than 99% of the vertebrate ocular ...', 
    # 'emb_scores': [0.87, 0.86, 0.85, 0.84, 0.87]}
    # ) (input_keys={'question', 'text_info', 'emb_scores', 'relation_info'})
