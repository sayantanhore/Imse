import json
import re

html_file = open("index_2.html", "r")

text = ""

for line in html_file.readlines():
    text = text + line + "\n"
#print text

colours = re.findall('#([a-zA-Z0-9]{6})', text)

print colours

with open("colours.txt", "w") as txt_file:
    for i in range(len(colours)):
        txt_file.write("'#" + str(colours[i]) + "', ")




'''
colour_dict = {}

for i in range(len(colours)):
    k = i / 25
    if not k in colour_dict:
        colour_dict[k] = [colours[i]]
    else:
        colour_dict[k].append(colours[i])

print colour_dict

with open("colors.json", "w") as json_file:
    json.dump(colour_dict, json_file)

'''