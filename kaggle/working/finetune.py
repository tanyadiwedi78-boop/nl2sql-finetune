import argparse
import json 
import random 
import re
import sys 
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT/ "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
ADAPTER_DIR = ROOT / "outputs" / "nl2sql-lora"
RESULTS_PATH = ROOT / "eval_results.json"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SCHEMA_PREAMBLE = (
    "You are and sql expert. Given the database schema below , write a single" 
    "SQL query that answers the questions.\n\n"
    "Schema:\n"
    "customer(customer_id , name , email , country , signup_date)\n"
    "products(product_id , name , category , price)\n"
    "orders(order_id , customer_id , order_date , status)\n"
    "order_items(order_item_id , order_id , product_id , quantity)\n"
    
)

CATEGORIES = ["Electronics" , "Clothing" , "Home & Kitchen" , "Books" , "Toys" , "Sports"]
COUNTRIES = ["USA" , "Canada" , "India" , "UK" , "Germany" , "Brazil" , "France" , "Japan"]
STATUSES = ["pending" , "shipped" , "delivery" , "cancelled"]

#===========================================================================
# DATASET GENERATION
#===========================================================================
# Template based Natural language - SQL Pairs . Fully reproductile (seeded) , no external.
# API needed . Swap it out your own data/doamin and evreything else
# below (train / ask / evaluate ) keeps working unchanged

def _sample_int(rng , lo , hi):
    return rng.randint(lo , hi)


def _gen_simple_filter(rng):
    country = rng.choice(COUNTRIES)
    q = f"List the names and emails of all customers from {country}"
    sql = f"SELECT name , email FROM customer WHERE country = '{country}';"
    return q , sql

def _gen_count(rng):
    status = rng.choice(STATUSES)
    q = f"How many orders have the status '{status}'"
    sql = f"SELECT COUNT(*) FROM orders WHERE status = '{status}';"
    return q ,sql

def _gen_avg_price(rng):
    cat = rng.choice(CATEGORIES)
    q = f"What is the average price of products in the {cat} category ?"
    sql = f"SELECT AVG(price) FROM products WHERE category "
    return q , sql

def _gen_top_n(rng):
    n = sample_int(rng , 3 , 10)
    q = f"Show the top {n} most expensive products."
    sql = f"SELECT name , price FROM products ORDER BY price DESC LIMIT {n};"
    return q , sql

def _gen_join_customer_orders(rng):
    n = _sample_int(rng , 3, 10)
    q = f"Show the top {n} customers by total amount spent."
    sql = (
        "SELECT c.name , SUM(p.price * oi.quantity) AS total_spent"
        "FROM customers c"
        "JOIN orders o ON c.customer_id = o.customer_id"
        "JOIN order_items oi ON o.order_id = oi.order_id"
        "JOIN products p ON oi.product_id = p.product_id"
        "GROUP BY c.customer_id , c.name "
        f"ORDER BY total_spent DESC LIMIT {n}"
    )
    return q , sql 
    

def _gen_group_by_category(rng):
    q = "Show the total revenue for each product category."
    sql = (
        "SELECT p.category, SUM(p.price * oi.quantity) AS revenue "
        "FROM products p "
        "JOIN order_items oi ON p.product_id = oi.product_id "
        "GROUP BY p.category;"
    )
    return q, sql


def _gen_status_filter(rng):
    status = rng.choice(STATUSES)
    country = rng.choice(COUNTRIES)
    q = f"List order IDs with status '{status}' placed by customers from {country}."
    sql = (
        "SELECT o.order_id FROM orders o "
        "JOIN customers c ON o.customer_id = c.customer_id "
        f"WHERE o.status = '{status}' AND c.country = '{country}';"
    )
    return q, sql


def _gen_date_filter(rng):
    year = _sample_int(rng, 2021, 2025)
    q = f"How many orders were placed in {year}?"
    sql = f"SELECT COUNT(*) FROM orders WHERE strftime('%Y', order_date) = '{year}';"
    return q, sql


def _gen_revenue_per_product(rng):
    n = _sample_int(rng, 3, 8)
    q = f"Show the top {n} products by total revenue generated."
    sql = (
        "SELECT p.name, SUM(p.price * oi.quantity) AS revenue "
        "FROM products p "
        "JOIN order_items oi ON p.product_id = oi.product_id "
        "GROUP BY p.product_id, p.name "
        f"ORDER BY revenue DESC LIMIT {n};"
    )
    return q, sql


def _gen_customers_no_orders(rng):
    q = "List customers who have never placed an order."
    sql = (
        "SELECT c.name FROM customers c "
        "LEFT JOIN orders o ON c.customer_id = o.customer_id "
        "WHERE o.order_id IS NULL;"
    )
    return q, sql


def _gen_price_threshold(rng):
    price = _sample_int(rng, 10, 500)
    cat = rng.choice(CATEGORIES)
    q = f"List products in the {cat} category that cost more than ${price}."
    sql = f"SELECT name, price FROM products WHERE category = '{cat}' AND price > {price};"
    return q, sql


def _gen_count_by_country(rng):
    country = rng.choice(COUNTRIES)
    q = f"How many customers are there from {country}?"
    sql = f"SELECT COUNT(*) FROM customers WHERE country = '{country}';"
    return q, sql


def _gen_recent_signups(rng):
    year = _sample_int(rng, 2021, 2025)
    q = f"List customers who signed up in {year}."
    sql = f"SELECT name, signup_date FROM customers WHERE strftime('%Y', signup_date) = '{year}';"
    return q, sql


def _gen_min_max_price(rng):
    cat = rng.choice(CATEGORIES)
    q = f"What is the most expensive product in the {cat} category?"
    sql = f"SELECT name, price FROM products WHERE category = '{cat}' ORDER BY price DESC LIMIT 1;"
    return q, sql


def _gen_orders_per_customer(rng):
    n = _sample_int(rng, 1, 10)
    q = f"List customers who have placed more than {n} orders."
    sql = (
        "SELECT c.name, COUNT(o.order_id) AS order_count FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.customer_id, c.name "
        f"HAVING COUNT(o.order_id) > {n};"
    )
    return q, sql


def _gen_category_count(rng):
    q = "How many products are there in each category?"
    sql = "SELECT category, COUNT(*) FROM products GROUP BY category;"
    return q, sql


def _gen_cheapest_n(rng):
    n = _sample_int(rng, 3, 10)
    q = f"Show the {n} cheapest products."
    sql = f"SELECT name, price FROM products ORDER BY price ASC LIMIT {n};"
    return q, sql


def _gen_customers_by_status_count(rng):
    status = rng.choice(STATUSES)
    q = f"How many distinct customers have an order with status '{status}'?"
    sql = (
        "SELECT COUNT(DISTINCT c.customer_id) FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id "
        f"WHERE o.status = '{status}';"
    )
    return q, sql



GENERATORS = [
    _gen_simple_filter, _gen_count, _gen_avg_price, _gen_top_n,
    _gen_join_customer_orders, _gen_group_by_category, _gen_status_filter,
    _gen_date_filter, _gen_revenue_per_product, _gen_customers_no_orders,
    _gen_price_threshold, _gen_count_by_country, _gen_recent_signups,
    _gen_min_max_price, _gen_orders_per_customer, _gen_category_count,
    _gen_cheapest_n, _gen_customers_by_status_count,
]


def _to_record(question, sql):
    return {
        "instruction": SCHEMA_PREAMBLE + "\nQuestion: " + question,
        "input": "",
        "output": sql,
    }


def _dedupe(pairs):
    seen, out = set(), []
    for q, sql in pairs:
        key = (q, sql)
        if key not in seen:
            seen.add(key)
            out.append((q, sql))
    return out


def cmd_gen_data(args):
    rng = random.Random(args.seed)
    target = args.n_train + args.n_test

    unique = []
    multiplier = 6
    while len(unique) < target and multiplier < 200:
        raw = [rng.choice(GENERATORS)(rng) for _ in range(target * multiplier)]
        unique = _dedupe(raw)
        multiplier *= 2

    if len(unique) < target:
        print(f"WARNING: only {len(unique)} unique examples available "
              f"(requested {target}). Reduce --n-train/--n-test.")

    rng.shuffle(unique)
    train = unique[: args.n_train]
    test = unique[args.n_train: args.n_train + args.n_test]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        for q, sql in train:
            f.write(json.dumps(_to_record(q, sql)) + "\n")
    with open(TEST_PATH, "w") as f:
        for q, sql in test:
            f.write(json.dumps(_to_record(q, sql)) + "\n")

    print(f"Wrote {len(train)} training examples -> {TRAIN_PATH}")
    print(f"Wrote {len(test)} test examples -> {TEST_PATH}")


# =============================================================================
# 2. TRAINING  (QLoRA fine-tuning)
# =============================================================================

def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {"text": prompt + example["output"]}


def cmd_train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig , get_peft_model 
    from transformers import AutoModelForCasualLM , AutoTokenizer , BitsAndBytesConfig
    from trl import SFTTrainer , SFTConfig

    if not TRAIN_PATH.exists():
        sys.exit(f"No matching data at {TRAIN_PATH} . Run {Path(__file__).name} gen_data")

    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"Loading base model : {args.base_model} (4-bit = {use_cuda})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant  =True ,
            bnb_4bit_quant_type = "nf4" ,
            bnd_4bit_compute_dtype = torch.bfloat16 , 
        )

        model = AutoModelCasualLM.from_pretrained(
            args.base_model , quantization_config = bnd_config , device_map = "auto"
        )

    else:
        model = AutoModelCasualLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
            r = args.lora_r ,
            lora_alpha = args.lora_alpha ,
            target_modules = ["q_proj" , "k_proj" , "v_proj" , "o_proj"],
            lora_dropuout = 0.05,
            bias = "None",
            task_type = "CASUAL_LM",
        )
    model = get_peft_model(model , lora_config)
    model.print_trainable_parameters()

    records = load_jsonl(TRAIN_PATH)
    train_dataset = Dataset.drom_list(records).map(_format_example)
    print(f"Loaded {len(train_dataset)} training examples from {TRAIN_PATH}")

    sft_config = SFTConfig(
        output_dir = str(ADAPTER_DIR),
        num_train_epochs = args.epochs , 
        per_device_train_batch_size = args.batch_size , 
        gradient_accumulation_steps = 4,
        learning_rate = args.lr,
        logging_steps = 10,
        save_strategy = "epoch",
        bf16 = use_cuda,
        max_seq_length = args.max_seq_len,
        dataset_text_field = "text",
        report_to = "none",
        optim="adamw_8bit",
        )

    trainer = SFTTrainer(
        model = model , 
        args = sft_config,
        train_dataset = train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents= True , exist_ok = True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretarined(str(ADAPTER_DIR))
    print(f"\n Saved LoRA adapter to {ADAPTER_DIR}")


#============================================================================
# INFERENCE
#============================================================================

def build_prompt(question: str)-> str:
    return f"### Instruction:\n {SCHEMA_PREAMBLE}\nQuestion: {question}\n\n### Response:\n"


def _load_model(base_model_name:str,use_adapter=bool , cpu=bool):
    import torch
    from transformers import AutoModelForCasualLM , AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available() and not cpu
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )

    if use_adapter:
        from peft import PeftModel
        if not ADAPTER_DIR.exists():
            sys.exit(f"No adapter at {ADAPTER_DIR}. Run: python {Path(__file__).name} train")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    model.eval()
    return model, tokenizer


def _generate_sql(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    import torch
    prompt = _build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()


def cmd_ask(args):
    model, tokenizer = _load_model(args.base_model, use_adapter=not args.base_only, cpu=args.cpu)
    result = _generate_sql(model, tokenizer, args.question)
    label = "Base model" if args.base_only else "Fine-tuned model"
    print(f"\nQuestion: {args.question}\n{label} output:\n{result}\n")


# =============================================================================
# 4. EVALUATION 
# =============================================================================

def _normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").lower()
    return re.sub(r"\s+", " ", sql)


def _load_test_set(limit=None):
    if not TEST_PATH.exists():
        sys.exit(f"No test data at {TEST_PATH}. Run: python {Path(__file__).name} gen-data")
    records = _load_jsonl(TEST_PATH)
    out = []
    for r in records:
        question = r["instruction"].split("Question:")[-1].strip()
        out.append({"question": question, "gold_sql": r["output"]})
    return out[:limit] if limit else out


def _run_eval(model, tokenizer, test_set, label):
    correct, rows = 0, []
    for i, ex in enumerate(test_set):
        pred = _generate_sql(model, tokenizer, ex["question"])
        pred_sql = pred.split("\n")[0].strip()
        is_match = _normalize_sql(pred_sql) == _normalize_sql(ex["gold_sql"])
        correct += int(is_match)
        rows.append({"question": ex["question"], "gold_sql": ex["gold_sql"],
                    "predicted_sql": pred_sql, "match": is_match})
        print(f"[{label}] {i+1}/{len(test_set)}  match={is_match}")
    accuracy = correct / len(test_set) if test_set else 0.0
    return accuracy, rows


def cmd_evaluate(args):
    test_set = _load_test_set(args.limit)
    results = {}

    if args.which in ("base", "both"):
        print("\n=== Evaluating BASE model ===")
        model, tokenizer = _load_model(args.base_model, use_adapter=False, cpu=args.cpu)
        acc, rows = _run_eval(model, tokenizer, test_set, "base")
        results["base"] = {"accuracy": acc, "rows": rows}
        del model

    if args.which in ("finetuned", "both"):
        print("\n=== Evaluating FINE-TUNED model ===")
        model, tokenizer = _load_model(args.base_model, use_adapter=True, cpu=args.cpu)
        acc, rows = _run_eval(model, tokenizer, test_set, "finetuned")
        results["finetuned"] = {"accuracy": acc, "rows": rows}
        del model

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for key, val in results.items():
        print(f"{key}: exact-match accuracy = {val['accuracy']*100:.1f}%")
    print(f"\nFull results saved to {RESULTS_PATH}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-data", help="Generate the NL2SQL train/test dataset")
    p_gen.add_argument("--n-train", type=int, default=300)
    p_gen.add_argument("--n-test", type=int, default=50)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_gen_data)

    p_train = sub.add_parser("train", help="QLoRA fine-tune the base model")
    p_train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lora-r", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--max-seq-len", type=int, default=512)
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_ask = sub.add_parser("ask", help="Ask a question, get generated SQL")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p_ask.add_argument("--base-only", action="store_true")
    p_ask.add_argument("--cpu", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("evaluate", help="Evaluate base vs fine-tuned model on the test set")
    p_eval.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p_eval.add_argument("--which", choices=["base", "finetuned", "both"], default="both")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

        
