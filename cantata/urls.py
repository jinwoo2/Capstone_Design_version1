"""cantata URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.conf.urls import url, include
from django.contrib.auth import views as auth_views
from first import views as first_views

urlpatterns = [

    path('', include('first.urls')),
    path('admin/', admin.site.urls),
    #url(r'^accounts/signup$', first_views.CreateUserView.as_view(), name = 'signup'),
    #url(r'^accounts/login/done$', first_views.RegisteredView.as_view(), name = 'create_user_done'),
]
