"""
URL configuration for resume_analyzer project.
"""

from django.contrib import admin
from django.urls import path
from analyzer import views


urlpatterns = [

    path(
        'admin/',
        admin.site.urls
    ),

    path(
        '',
        views.home,
        name='home'
    ),

    path(
        'rewrite/',
        views.rewrite_resume,
        name='rewrite_resume'
    ),

]