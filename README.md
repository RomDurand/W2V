# W2V
Une étude sur certain paramètres de l'algorithme word2Vec.

La méthode prep_data_tp2TL est la méthode qui regroupe toutes les fonction de pre-traitement du corpus. 
Elle possède une méthode build_data qui prend en paramètre alpha et t et qui construit les objets nécessaire à l'apprentissage et au test.

La méthode Alias_methode contient l'implémentation de la méthode Alias qui est une methode d'échantillonage en temps constant.

Le méthode w2v contient principalement la fonction de constuction des exemples d'apprentissages et la fonction d'apprentissage. 
Le choix des hyper-paramètres et le lancement de l'apprentissage se fait dans la fonction w2v.
