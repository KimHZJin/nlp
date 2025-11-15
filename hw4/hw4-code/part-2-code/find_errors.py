import re

pred = open('results/t5_ft_ft_experiment_dev.sql').readlines()
gold = open('data/dev.sql').readlines()
nl = open('data/dev.nl').readlines()

def normalize(sql):
    # Remove whitespace differences
    return re.sub(r'\s+', ' ', sql).strip()

real_errors = []
for i in range(len(pred)):
    if normalize(pred[i]) != normalize(gold[i]):
        real_errors.append((i, nl[i].strip(), gold[i].strip()[:200], pred[i].strip()[:200]))

print(f"Total real errors: {len(real_errors)}/{len(pred)}")
for idx, n, g, p in real_errors[:5]:
    print(f"\n=== Error {idx} ===")
    print(f"NL: {n}")
    print(f"GOLD: {g}")
    print(f"PRED: {p}")
