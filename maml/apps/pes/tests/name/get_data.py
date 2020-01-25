from pymacy.db import get_db
from bson.json_util import dumps

db = get_db()

results = []
count = 0
for i in db.benchmark.find({"element": "Ni"}):
    count += 1
    if count > 100:
        break
    results.append(i)
print(results[0])
with open("Ni.json", 'w') as f:
    file = dumps(results)
    f.write(file)
