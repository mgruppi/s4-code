

# fin = open("results_param_search_semeval.txt")
#
# for line in fin:
#     l = line.replace("|s4", "")
#     l = l.lstrip("|").rstrip("|\n")
#     print(l)
# fin.close()


fin = open("results_param_search_ukus.txt")


for line in fin:
    x = line.strip().strip("|").split("|")
    cls_method = x[0]
    align_method = x[1]
    accuracy = x[2].split("+-")[0]
    prec = x[3].split("+-")[0]
    recall = x[4].split("+-")[0]
    f1 = x[5].split("+-")[0]
    if len(x) > 6:
        r = x[6]

    print(cls_method, align_method, accuracy, prec, recall, f1, r, sep="|")

fin.close()
