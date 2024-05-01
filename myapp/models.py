from django.db import models

# Create your models here.

class Login(models.Model):
    username=models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    type = models.CharField(max_length=100)

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    dob = models.DateField()
    phonenumber = models.BigIntegerField()
    place = models.CharField(max_length=100)
    gender = models.CharField(max_length=100)
    LOGIN=models.ForeignKey(Login,on_delete=models.CASCADE)

class Complaint(models.Model):
    complaint =models.CharField(max_length=100)
    USER = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=100)
    date = models.DateField()
    reply = models.CharField(max_length=100)

class Review(models.Model):
    review=models.CharField(max_length=100)
    USER = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()

class Diary(models.Model):
    USER = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.CharField(max_length=300)
    date = models.DateField()
    emotion=models.CharField(max_length=100,default="")

class FaceEmotion(models.Model):
    USER = models.ForeignKey(User, on_delete=models.CASCADE)
    photo = models.CharField(max_length=500)
    date = models.DateField()
    emotion=models.CharField(max_length=500,default="")

class Tips(models.Model):
    USER = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    tips = models.CharField(max_length=500)

class Mentoringclass(models.Model):
    date = models.DateField()
    link = models.CharField(max_length=500)








