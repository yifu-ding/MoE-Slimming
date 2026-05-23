# lm-eval==0.3.0

import json
import os
from typing import List

from lm_eval import tasks as lm_eval_tasks
from lm_eval import evaluator

def eval_tasks(model, model_name, tokenizer, tasks: List[str], limit=10, max_seqlen=2048, batch_size=1, num_fewshot=0, verbose=False):
    if tasks == list() or tasks == ['']:
        return dict()
    
    if  'gsm8k' in tasks or 'gsm8k_cot' in tasks or 'gsm8k_cot_fs' in tasks or 'human_eval' in tasks or "humaneval" in tasks:
        from .eval_utils_lm import LMEvalAdaptor
        print("Using  eval_utils_lm for gsm8k or humaneval")
        import os
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    else:
        from .eval_utils_lm_2 import LMEvalAdaptor
        print("Using eval_utils_lm_2")

    lm_eval_model = LMEvalAdaptor(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seqlen
    )
    tm = lm_eval_tasks.TaskManager()  

    print(f"Evaluating tasks: {tasks}, limit: {limit}, max_seqlen: {max_seqlen}, batch_size: {batch_size}, num_fewshot: {num_fewshot}", flush=True)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        task_manager=tm, 
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit if limit > 0 else None, 
        confirm_run_unsafe_code=True
    )
    result = results['results']
    # print(result)
    return result


def _get_num_fewshot(tasks):
    if 'gsm8k_cot_fs' in tasks:
        return 8
    elif 'wikitext2' in tasks:
        return 0
    elif "mmlu" in tasks:
        return 5
    else:
        return 0
    

def eval_fn(args, model, tokenizer, remain_tasks, verbose=False):
    
    model_name = args.model_name_or_path
    max_seqlen = getattr(args, "eval_max_len", getattr(args, "max_seq_length", 2048))
    tasks = remain_tasks
    batch_size = getattr(args, 'batch_size', 1)
    num_fewshot = getattr(args, 'num_fewshot', _get_num_fewshot(tasks))
    limit = args.eval_sample_limit

    if 'wikitext2' in tasks:
        tasks.remove('wikitext2')
        tasks.append('wikitext')
    if 'gsm8k_cot_fs' in tasks:
        tasks.remove('gsm8k_cot_fs')
        tasks.append('gsm8k_cot')

    _results = eval_tasks(model, model_name, tokenizer, tasks, limit, max_seqlen, batch_size, num_fewshot=num_fewshot, verbose=verbose)
    print(_results)

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, "lm_harness")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.eval_task_names}-fs{num_fewshot}-results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")

    return _results

