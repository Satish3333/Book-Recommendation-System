{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 250,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(271360, 8)\n",
      "(1149780, 3)\n",
      "(278858, 3)\n",
      "Animal Farm\n",
      "The Handmaid's Tale\n",
      "Brave New World\n",
      "The Vampire Lestat (Vampire Chronicles, Book II)\n",
      "The Hours : A Novel\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "books=pd.read_csv('Books.csv')\n",
    "users=pd.read_csv('Users.csv')\n",
    "ratings=pd.read_csv('Ratings.csv')\n",
    "books.head()\n",
    "ratings.head()\n",
    "print(books.shape)\n",
    "print(ratings.shape)\n",
    "print(users.shape)\n",
    "users.isnull().sum()\n",
    "books.isnull().sum()\n",
    "ratings.isnull().sum()\n",
    "books.duplicated().sum()\n",
    "ratings.duplicated().sum()\n",
    "users.duplicated().sum()\n",
    "#Popularity based recommender system\n",
    "ratings_with_name=ratings.merge(books,on='ISBN')\n",
    "ratings_with_name\n",
    "num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()\n",
    "num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)\n",
    "num_rating_df\n",
    "avg_rating_df=ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()\n",
    "avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace=True)\n",
    "avg_rating_df\n",
    "popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')\n",
    "popular_df\n",
    "popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)\n",
    "popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]\n",
    "popular_df\n",
    "##collaborative filtering based recommender system\n",
    "x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200\n",
    "k_users=x[x].index\n",
    "k_users.shape\n",
    "filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(k_users)]\n",
    "y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50\n",
    "famous_books=y[y].index\n",
    "famous_books\n",
    "final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]\n",
    "pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')\n",
    "pt\n",
    "pt.fillna(0,inplace=True)\n",
    "pt\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "similarity_score=cosine_similarity(pt)\n",
    "similarity_score\n",
    "def recommend(book_name):\n",
    "    index = np.where(pt.index==book_name)[0][0]\n",
    "    similar_items= sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]\n",
    "    for i in similar_items:\n",
    "        print(pt.index[i[0]])\n",
    "recommend('1984')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
