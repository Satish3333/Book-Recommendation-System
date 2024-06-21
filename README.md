import pandas as pd
import numpy as np
books=pd.read_csv('Books.csv')
users=pd.read_csv('Users.csv')
ratings=pd.read_csv('Ratings.csv')
books.head()
ratings.head()
print(books.shape)
print(ratings.shape)
print(users.shape)
users.isnull().sum()
books.isnull().sum()
ratings.isnull().sum()
books.duplicated().sum()
ratings.duplicated().sum()
users.duplicated().sum()
#Popularity based recommender system
ratings_with_name=ratings.merge(books,on='ISBN')
ratings_with_name
num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df
avg_rating_df=ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace=True)
avg_rating_df
popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df
popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)
popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]
popular_df
##collaborative filtering based recommender system
x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
k_users=x[x].index
k_users.shape
filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(k_users)]
y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index
famous_books
final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')
pt
pt.fillna(0,inplace=True)
pt
from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(pt)
similarity_score
def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_items= sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    for i in similar_items:
        print(pt.index[i[0]])
recommend('1984')
