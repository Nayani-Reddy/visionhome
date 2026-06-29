with open("../backend/furniture_tags.json", "r") as f:
    TAGS = json.load(f)

def search(text):
    text = text.lower()
    results = []

    for img, tags in TAGS.items():
        score = 0
        for tag in tags:
            if tag in text:
                score += 1
        if score > 0:
            results.append((img, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [f"/assets/furniture/{r[0]}" for r in results[:5]]


