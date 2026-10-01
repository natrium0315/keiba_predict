def combinations(data):
    

if __name__ == "__main__":
    with open("result.txt", "r", encoding = "utf-8") as f:
        first_flag = True
        race = None
        data = []
        result = []

        for line in f:
            text = line.split(":")
            tag = text[0]
            
            if tag == "レース":
                if race != None:
                    combination = combinations(data)
                    result.append([race, combination])
                    data.clear()
                race = text[1]
            if tag == "予測":
                number = text[1]
                name = text[2]
                score = float(text[3])
                data.append((number, name, score))

    if race is not None:
        combination = combinations(data)
        result.append([race, combination])
