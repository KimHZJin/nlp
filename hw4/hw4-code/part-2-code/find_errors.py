import re

pred = open('results/t5_ft_ft_experiment_dev.sql').readlines()
gold = open('data/dev.sql').readlines()
nl = open('data/dev.nl').readlines()

def normalize(sql):
    return re.sub(r'\s+', ' ', sql).strip()

real_errors = []
for i in range(len(pred)):
    if normalize(pred[i]) != normalize(gold[i]):
        real_errors.append((i, nl[i].strip(), gold[i].strip(), pred[i].strip()))

# Count actual semantic differences (not just whitespace)
semantic_errors = []
for idx, n, g, p in real_errors:
    # Check if difference is more than just comma spacing
    if g.replace(' ,', ',') != p.replace(' ,', ','):
        semantic_errors.append((idx, n, g, p))

print(f"Semantic errors: {len(semantic_errors)}/{len(pred)}")
print(f"Whitespace-only: {len(real_errors) - len(semantic_errors)}/{len(pred)}")

for idx, n, g, p in semantic_errors[:10]:
    print(f"\n=== Error {idx} ===")
    print(f"NL: {n}")
    print(f"GOLD: {g[:300]}")
    print(f"PRED: {p[:300]}")
