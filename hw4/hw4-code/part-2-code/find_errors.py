import difflib

pred = open('results/t5_ft_ft_experiment_dev.sql').readlines()
gold = open('data/dev.sql').readlines()
nl = open('data/dev.nl').readlines()

errors = []
for i in range(len(pred)):
    # Normalize whitespace
    p = ' '.join(pred[i].split())
    g = ' '.join(gold[i].split())
    
    if p != g:
        errors.append({
            'idx': i,
            'nl': nl[i].strip(),
            'gold': g[:150],  # First 150 chars
            'pred': p[:150]
        })

# Show first 10 errors
for err in errors[:10]:
    print(f"\n=== Error {err['idx']} ===")
    print(f"NL: {err['nl']}")
    print(f"GOLD: {err['gold']}")
    print(f"PRED: {err['pred']}")
