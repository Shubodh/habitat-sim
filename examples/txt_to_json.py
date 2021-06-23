import json

filename = 'xview_poses-replica-apartment_0_run1.txt'

dict1 = {'position':[], 'rotation':[]}

with open(filename) as fh:
    l = 1
    for line in fh:
        description = list(line.strip().split(None, 4))
        #print(description)
        if l%2 != 0:
            dict1[description[0]].append([description[1], description[2], description[3]])
        else:
            dict1[description[0]].append([description[1], description[2], description[3], description[4]])
        l = l + 1

out_file = open("xview_poses-replica-apartment_0_run1.json", "w")
json.dump(dict1, out_file, indent=4, sort_keys=False)
out_file.close()
