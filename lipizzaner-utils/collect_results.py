import os
import re

directory_path = './log/'

directory = os.fsencode(directory_path)

from_time = ""

without_time_label = 0
searched_files = 1
matches = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)

    #is_master_file = re.search('lipizzaner', filename)

    #if is_master_file:

    f = open(directory_path + filename)
    lines = f.readlines()


    # Search for time label
    # time_label = re.search('(.*\d).* - lipizzaner_master', lines[0])
    # if not time_label:
    #     print("Searched file " + str(searched_files) + ": " + "TIME LABEL NOT FOUND - " + filename)
    #     without_time_label +=1
    #     searched_files += 1


    fid = "None"

    for line in lines:

        # Find values
        match = re.search('INFO - lipizzaner_master - Best result: .* \((.*),(.*)\)', line)

        if match:
            matches += 1
            fid = match.group(1)

    print("Searched file " + str(searched_files) + ": " + fid)
    searched_files += 1

print('***********************************************')
print("Total files searched: " + str(searched_files - 1))
print("Total best result lines found: " + str(matches))
print('***********************************************')
