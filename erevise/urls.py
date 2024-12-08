"""erevise URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import re_path as url
from . import views

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

urlpatterns = [
    url(r'^$', views.index),
    # path('hello/', views.hello),
    # path('runoob/', views.runoob),
    path('index/', views.index),
    # path('add_data_user/', views.add_data_user),
    # path('add_data_prompt/', views.add_data_prompt),
    # path('add_data_feedback/', views.add_data_feedback),
    # path('get_data_prompt/', views.get_data_prompt),
    # path('add_data_essay/', views.add_data_essay),
    # path('get_data_essay/', views.get_data_essay),
    path('admin/', admin.site.urls),
    # path('', views.home, name='home'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    # path('signup/', views.signup, name='signup'),
    # path('profile/', views.profile, name='profile'),
    path('roster/', views.roster, name='roster'),
    path('mvp/', views.submit_mvp_v4, name='submit_mvp'),
    path('mvp/article/', views.show_mvp, name='view_pdf'),
    path('space/', views.submit_space_v3, name='submit_space'),
    path('space/article/', views.show_space, name='show_space'),
    path('upload/', views.upload_file_v2, name='upload_file'),
    path('confirm/', views.get_user_info, name='confirm_user_info'),
    path('submission/', views.submission, name='submission'),
    path('process/', views.process_essays_v2, name='process'),
    # path('save_feedback/<str:user_name>/', views.save_feedback, name='save_feedback'),
    path('save_feedback/', views.save_feedback, name='save_feedback'),
]