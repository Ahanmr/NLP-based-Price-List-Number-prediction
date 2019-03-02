from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
corpus = ["ordinary, soap, weird",
        "ordinary, loop, soap",
        "food, pantry, car",
        "ticket,pantry,passenger",
         "ordinary,soap,weird",
         "ordinary,towel,normal",
         "food,pantry,car",
         "ordinary, loop1, soap, random"]
db_add=input("Enter a new price list(PL) entry: ")
corpus.append(str(db_add))
vectorizer=CountVectorizer()
features=vectorizer.fit_transform(corpus).todense()
print(features)
print(vectorizer.vocabulary_)
for f in features:
    print(euclidean_distances(features[0],f))

