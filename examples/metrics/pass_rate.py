from bigcodebench.eval import untrusted_check


def pass_rate(code, unit_tests, entry_point='task_func'):
    result = untrusted_check(code,
                            unit_tests,
                            entry_point,
                            max_as_limit=300*1024,
                            max_data_limit=300*1024,
                            max_stack_limit=300*1024,
                            min_time_limit=2,
                            gt_time_limit=5)
    print(result)
    return float(result[0] == 'pass')
