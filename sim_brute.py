import math
import itertools

def test_wrapper(raw_pw_stream):
    history = [0,0,0]
    fill = 0
    
    medians = []
    # Simulate the control loop reading the stream
    for raw in raw_pw_stream:
        slot = fill if fill < 3 else (fill % 3)
        history[slot] = raw
        fill += 1
        
        if fill >= 3:
            s_hist = sorted(history)
            med = s_hist[1]
            medians.append(med)
            
    # Now simulate multi-turn
    if not medians: return 0
    last = medians[0]
    revs = 0
    for med in medians[1:]:
        delta = med - last
        if delta > math.pi: revs -= 1
        if delta < -math.pi: revs += 1
        last = med
        
    return revs

# Let's see if ANY sequence of length 6 using only two or three values can runaway
values = [0.1, 3.1, 6.1]

for p in itertools.product(values, repeat=10):
    revs = test_wrapper(p)
    if abs(revs) >= 1:
        print("Runaway sequence:", p, "Revs:", revs)
        break

print("Done")
