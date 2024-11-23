import torch

config = {
  'category_label_num': 3000,#[2,2975] total:294
  'sentiment_label_dict': {'Negative':0,'Neutral':1,'Positive':2},
  'article_type_dict': {
    'article_default': 0,
    'article_webtv': 1,
    'article_page_nine_girl': 2,
    'article_questions_and_answers': 3,
    'article_feature': 4,
    'article_opinionen': 5,
    'article_native': 6,
    'article_scribblelive': 7,
    'article_fullscreen_gallery': 8,
    'article_editorial_production': 9,
    'article_standard_feature': 10,
    'article_native_feature': 11,
    'article_accordion': 12,
    'article_video_standalone': 13,
    'article_image_gallery': 14,
    'article_timeline': 15},
  'news_vector':300,
  'news_pca_vector':64,

  'news_feature':64,
  'history_max_length': 200,#line

  'datetime_normalize_length':365*4,#day
  'read_time_normalize_length':45,#min set_max:30min

  'total_inview_normalize_length':1e9,#

}
