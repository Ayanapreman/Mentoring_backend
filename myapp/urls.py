"""
URL configuration for Mentoring project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path, include

from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.login),
    path('changepwd/',views.changepwd),
    path('viewcmplaint/', views.viewcmplaint),
    path('sendreply/<id>',views.sendreply),
    path('viewreview/', views.viewreview),
    path('adminhome/',views.adminhome),
    path('loginpost/',views.loginpost),
    path('changepwdpost/',views.changepwdpost),
    path('viewcmplaintpost/',views.viewcmplaintpost),
    path('sendreplypost/',views.sendreplypost),
    path('viewreviewpost/',views.viewreviewpost),
    path('logout/',views.logout),
    path('viewemotiongraph/<id>', views.viewemotiongraph),
    path('viewuser/', views.viewuser),
    path('viewemotiongraph_post/', views.viewemotiongraph_post),
    path('sendtips/<id>', views.sendtips),
    path('tipspost/', views.tipspost),
    path('sendclass/', views.sendclass),
    path('classpost/', views.classpost),
    path('forgotpwd/', views.forgotpwd),
    path('forgotpwd_post/', views.forgotpwd_post),




    path('user_login/', views.user_login),
    path('user_register/',views.user_register),
    path('user_viewprofile/', views.user_viewprofile),
    path('user_editprofile/',views.user_editprofile),
    path('user_adddiary/',views.diarywriting),
    path('user_sendcmplnt/',views.user_sendcmplnt),
    path('user_viewreply/',views.user_viewreply),
    path('user_sendreview/',views.user_sendreview),
    path('user_changepwd/',views.user_changepwd),
    path('user_viewdiary/',views.user_viewdiary),
    path('user_diarycam/', views.user_diarycam),
    path('user_viewtips/', views.user_viewtips),
    path('user_mentoringclass/', views.user_mentoringclass),
    path('user_forgotpwd/', views.user_forgotpwd),


    path('ConfusionMatrix/', views.ConfusionMatrix),



]
