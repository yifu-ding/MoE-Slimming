
def _get_lm_harness_tasks():
    return ["human_eval", "humaneval", "gsm8k", "gsm8k_cot", "gsm8k_cot_fs", "c4", "wikitext2", "wikitext", "commonsenseqa", "mmlu", "hellaswag", "piqa", "boolq", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]


def eval_dispatch(args, model, tokenizer, verbose=False):
    _tasks = [task for task in args.eval_task_names.split(",")]
    results = {}
    if any(task.lower() in _get_lm_harness_tasks() for task in _tasks): 
        from eval.lm_harness.eval import eval_fn
        res = eval_fn(args, model, tokenizer, _tasks, verbose=verbose)
        results.update(res)

    return results
