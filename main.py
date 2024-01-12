import pandas as pd
import numpy as np

import os 
import Recommenders



song_1 = pd.read_csv('triplets_file.csv')
#print (song_1.shape)
#print (song_1.head())

#Load the Data - song_data
song_2 = pd.read_csv('song_data.csv')
#print(song_2.shape)
#display(song_2.head())

#how many unique users are there
#print(len(song_1.user_id.unique()))

#left join on df1 and df2 and shortening the size
song_df = pd.merge(song_1, song_2.drop_duplicates(['song_id']), on='song_id', how='left').sample(frac=1)
song_df = song_df.head(int(len(song_df.user_id)/2))
#display (song_df.shape)
#display (song_df.head())
#display (song_df.tail())

#shorten the size of the data
#song_df=song_df.head(1000000)
#display(song_df)

#add new column
song_df['song'] = song_df['title']+' - '+song_df['artist_name']

#new dataframe for grouping and handling
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()

grouped_sum = song_grouped['listen_count'].sum()

song_grouped['percentage'] = (song_grouped['listen_count']/grouped_sum)*10000
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])

#popularity 
pr = Recommenders.popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')

#Display the top 10 popular songs- User 5 FOR ALL USERS IT'S THE SAME
print("The top songs of the decade is:")
#print(pr.recommend(song_df['user_id'][0]))

#in terms of item similarity
ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')

a = int(input("Enter the id index of the user for who you need recommendation:"))
user_items = ir.get_user_items(song_df['user_id'][a])

print("user no ",a," song history")
# display user songs history
for user_item in user_items:
    print(user_item)

print()
print("song recommended for user no.: ",a)
result = display(ir.recommend(song_df['user_id'][a]))
for z in result:
    print(z)
