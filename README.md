# Projet 6 - Classifiez automatiquement des biens de consommation

Openclassrooms parcours **Data Scientist**

## Problématique

l’entreprise "**Place de marché**" souhaite lancer une marketplace e-commerce.

Sur la place de marché, des vendeurs proposent des articles à des acheteurs en postant une photo et
une description.

Pour l'instant, l'attribution de la catégorie d'un article est effectuée manuellement par les
vendeurs, et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit.

Pour rendre l’expérience utilisateur des vendeurs (faciliter la mise en ligne de nouveaux articles)
et des acheteurs (faciliter la recherche de produits) la plus fluide possible, et dans l'optique
d'un passage à l'échelle, **il devient nécessaire d'automatiser cette tâche**.

## Les données

Un premier jeu de données d’articles avec la photo et une description associée :
[le lien pour télécharger](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip)

## Mission

Etudier la faisabilité d'un **moteur de classification** des articles en différentes catégories,
avec un niveau de précision suffisant.

- **réaliser une première étude de faisabilité d'un moteur de classification**, d'articles, basé sur
  une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.

- **analyser le jeu de données** en **réalisant un prétraitement** des descriptions des produits et
  des images, une **réduction de dimension**, puis un **clustering**, à présenter sous la forme de
  graphiques en deux dimensions, et confirmés par un calcul de similarité entre les catégories
  réelles et les clusters.
- illustre que les caractéristiques extraites permettent de regrouper des produits de même
  catégorie.

- démontrer, par cette approche de modélisation, la faisabilité de regrouper automatiquement des
  produits de même catégorie

## Plan du projet

Le plan de ce projet ce trouve en plus de détail dans le document
[project_plan.md](./project_plan.md).

## Livrables de ce projet

### [P6_01_textes_clustering.ipynb](./P6_02_textes_clustering.ipynb)

Import des données textes

- Nettoyage (tokenization, stemming, lemmatisation)

Extraction des features textes, via la mise en œuvre :

- deux approches de type “bag-of-words”, comptage simple de mots et Tf-idf ;
- une approche de type word/sentence embedding classique avec Word2Vec (ou Glove ou FastText) ;
- une approche de type word/sentence embedding avec BERT ;
- une approche de type word/sentence embedding avec USE (Universal Sentence Encoder).

### [P6_02_images_clustering.ipynb](./P6_03_images_clustering.ipynb)

Nettoyage

- Filtrer le bruit
- Egaliser l'histogramme

Extraction des features images, via la mise en œuvre :

- un algorithme de type SIFT / ORB / SURF (via reduction de dimensions PCA/t-SNE);
- un algorithme de type CNN Transfer Learning.

### [P6_03_support.pptx](./P6_03_support.pptx)

- Présentation et conclusion

## Compétences évaluées

- [ ] Prétraiter des données image pour obtenir un jeu de données exploitable
- [ ] Représenter graphiquement des données à grandes dimensions
- [ ] Prétraiter des données texte pour obtenir un jeu de données exploitable
- [ ] Mettre en œuvre des techniques de réduction de dimension
