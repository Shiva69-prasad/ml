import csv
def find_s(f):
    d = list(csv.reader(open(f)))[1:]
    h = d[0][:-1]
    for r in d[1:]:
        if r[-1].lower()=='yes': h = [x if x==y else '?' for x,y in zip(h,r[:-1])] # lower is a function use ()
    return h
print(find_s('training_data.csv'))
