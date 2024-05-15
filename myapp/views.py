import smtplib

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.
from myapp.models import *


def login(request):
    return render(request,'loginindex.html')

def loginpost(request):
    username=request.POST['textfield']
    password=request.POST['textfield2']
    lobj=Login.objects.filter(username=username,password=password)
    if lobj.exists():
        lobjj=Login.objects.get(username=username,password=password)
        request.session['lid']=lobjj.id
        if lobjj.type =='admin':
            return HttpResponse('''<script>alert('success');window.location='/myapp/adminhome/'</script>''')
        else:
            return HttpResponse('''<script>alert('Try Again');window.location='/myapp/login/'</script>''')
    else:
        return HttpResponse('''<script>alert('invalid.Try Again');window.location='/myapp/login/'</script>''')



def changepwd(request):
    if request.session['lid'] == '':
        return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')
    else:
       return render(request,'changepwd.html')

def changepwdpost(request):
    currentpassword=request.POST['textfield']
    newpassword=request.POST['textfield2']
    confirmpassword = request.POST['textfield3']
    lobj = Login.objects.filter(id=request.session['lid'], password=currentpassword)
    if lobj.exists():
        lobjj = Login.objects.get(id=request.session['lid'], password=currentpassword)
        lobjj.password=newpassword
        lobjj.save()
        return HttpResponse('''<script>alert('success');window.location='/myapp/login/'</script>''')
    else:
        return HttpResponse('''<script>alert('incorrect password');window.location='/myapp/changepwd/'</script>''')


def forgotpwd(request):
    return  render(request,'forgotpassword.html')

def forgotpwd_post(request):
    username = request.POST['textfield']
    lobj = Login.objects.filter(username=username)
    if lobj.exists():
        import random
        new_pass = random.randint(0000, 9999)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("safedore3@gmail.com", "yqqlwlyqbfjtewam")  # App Password
        to = username
        subject = "Test Email"
        body = "Your new password is " + str(new_pass)
        msg = f"Subject: {subject}\n\n{body}"
        server.sendmail("s@gmail.com", to, msg)
        # Disconnect from the server
        server.quit()
        ress = Login.objects.filter(username=username).update(password=new_pass)
        return HttpResponse('''<script>alert('New password generated.please check your email...');window.location='/myapp/login/'</script>''')
    else:
        return HttpResponse('''<script>alert('Invalid...');window.location='/myapp/login/'</script>''')



def viewcmplaint(request):
    if request.session['lid'] == '':
        return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')
    else:
        res=Complaint.objects.all()
        return render(request,'Viewcmplaint.html',{'data':res})

def viewcmplaintpost(request):
    fromdate=request.POST['textfield']
    to = request.POST['textfield2']
    res=Complaint.objects.filter(date__range=[fromdate,to])
    return render(request,'Viewcmplaint.html',{'data':res})




def sendreply(request,id):

        res=Complaint.objects.get(id=id)
        return render(request,'sendReply.html',{'data':res})

def sendreplypost(request):
    reply=request.POST['textfield']
    did=request.POST['id1']
    obj=Complaint.objects.get(id=did)
    obj.reply=reply
    obj.status='Replied'
    obj.save()
    return HttpResponse('''<script>alert('Replied');window.location='/myapp/viewcmplaint/'</script>''')


def viewreview(request):
    if request.session['lid'] == '':
        return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')
    else:
        res=Review.objects.all
        return render(request,'ViewReview.html',{'data':res})

def viewuser(request):
    if request.session['lid'] == '':
        return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')
    else:
        res=User.objects.all
        return render(request,'viewUser.html',{'data':res})

def viewreviewpost(request):
    fromdate=request.POST['textfield']
    to = request.POST['textfield2']
    res = Review.objects.filter(date__range=[fromdate, to])
    return render(request,'ViewReview.html',{'data':res})

def adminhome(request):
    if request.session['lid']=='':
        return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')
    else:
        return render(request,'admin_home_index.html')

def logout(request):
    request.session['lid']=''
    return HttpResponse('''<script>alert('login required');window.location='/myapp/login/'</script>''')

def viewemotiongraph(request,id):
    emotion_dict = ["anger", "joy", "sadness", "fear","disgust","surprised","neutral"]

    s = []
    for i in emotion_dict:

        s.append(len(Diary.objects.filter(USER_id=id, emotion=i)))

    # print(s, emotion_dict)

    return render(request, "emotion.html", {'s': s, 'e': emotion_dict, 'id':id})



def viewfaceemotiongraph(request,id):
    emotion_dict = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad", "Surprised" ]

    s = []
    for i in emotion_dict:

        s.append(len(FaceEmotion.objects.filter(USER_id=id, emotion=i)))

    # print(s, emotion_dict)

    return render(request, "emotionface.html", {'s': s, 'e': emotion_dict, 'id':id})


def viewemotiongraph_face_post(request):
    from_ = request.POST['from_']
    to_ = request.POST['to_']
    id = request.POST['id']
    emotion_dict = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

    s = []
    for i in emotion_dict:
        s.append(len(FaceEmotion.objects.filter(USER_id=id, emotion=i,date__range=[from_,to_])))

    # print(s, emotion_dict)

    return render(request, "emotionface.html", {'s': s, 'e': emotion_dict, 'id': id})










def viewemotiongraph_post(request):
    from_ = request.POST['from_']
    to_ = request.POST['to_']
    id = request.POST['id']
    emotion_dict = ["anger", "joy", "sadness", "fear","disgust","surprised","neutral"]

    s = []
    for i in emotion_dict:

        s.append(len(Diary.objects.filter(USER_id=id, emotion=i,date__range=[from_,to_])))

    # print(s, emotion_dict)

    return render(request, "emotion.html", {'s': s, 'e': emotion_dict, 'id':id})

def sendtips(request, id):
    res = User.objects.get(id=id)
    return render(request, 'tips.html', {'data': res})


def tipspost(request):
    tip = request.POST['textfield']
    id = request.POST['id1']
    obj = Tips()
    obj.tips = tip
    obj.USER_id = id
    import datetime
    obj.date = datetime.date.today()
    obj.save()
    return HttpResponse('''<script>alert('success');window.location='/myapp/viewuser/'</script>''')


def sendclass(request):
    # res = .objects.get(id=id)
    return render(request, 'AddMentoringclass.html')


def classpost(request):
    link = request.POST['textfield']
    obj = Mentoringclass()
    obj.link = link
    import datetime
    obj.date = datetime.date.today()
    obj.save()
    return HttpResponse('''<script>alert('success');window.location='/myapp/sendclass/'</script>''')










def user_login(request):
    username = request.POST['name']
    password = request.POST['password']
    lobj = Login.objects.filter(username=username, password=password)
    if lobj.exists():
        lobjj = Login.objects.get(username=username, password=password)
        lid = lobjj.id
        if lobjj.type == 'user':
            return JsonResponse({'status': 'ok','lid':lid})

        else:
            return JsonResponse({'status': 'Not ok'})
    else:
        return JsonResponse({'status': 'Not ok'})


def user_register(request):
    name = request.POST['name']
    email = request.POST['email']
    dob = request.POST['dob']
    phonenumber = request.POST['phonenumber']
    place = request.POST['place']
    gender = request.POST['gender']
    password = request.POST['password']
    confirmpassword = request.POST['confirmpassword']

    if password==confirmpassword:
        reg = Login()
        reg.username=email
        reg.password=confirmpassword
        reg.type='user'
        reg.save()


        reg1=User()
        reg1.LOGIN=reg
        reg1.name=name
        reg1.email=email
        reg1.dob=dob
        reg1.phonenumber=phonenumber
        reg1.place=place
        reg1.gender=gender
        reg1.save()

        return JsonResponse({'status': 'ok'})
    else:
        return JsonResponse({'status': 'Not ok'})


def user_viewprofile(request):
    lid=request.POST['lid']
    profile=User.objects.get(LOGIN_id=lid)
    return JsonResponse({'status':'ok','name':profile.name,
                         'email':profile.email,'dob':profile.dob,
                         'phonenumber':profile.phonenumber,'place':profile.place,
                         'gender': profile.gender,})

def user_editprofile(request):
    lid=request.POST['lid']
    name = request.POST['name']
    email = request.POST['email']
    dob = request.POST['dob']
    phonenumber = request.POST['phonenumber']
    place = request.POST['place']
    gender = request.POST['gender']
    reg1 = User.objects.get(LOGIN_id=lid)
    reg1.name = name
    reg1.email = email
    reg1.dob = dob
    reg1.phonenumber = phonenumber
    reg1.place = place
    reg1.gender = gender
    reg1.save()

    return JsonResponse({'status': 'ok'})

def diarywriting(request):
    lid= request.POST["lid"]
    diarycontent= request.POST["content"]

    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model_path = r'C:\Users\dell\PycharmProjects\Mentoring\michellejieli'
    tokenizer_path = r'C:\Users\dell\PycharmProjects\Mentoring\michellejieli'

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    sentiment_analysis_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # # Use a pipeline as a high-level helper
    # from transformers import pipeline
    #
    # pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
    #
    #
    #
    # # diarycontent= "I love you so much"
    # #
    # # import torch
    # # from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    # # from torch.nn.functional import softmax
    # #
    # # # Load pre-trained DistilBERT model and tokenizer
    # # model_name = 'distilbert-base-uncased'
    # # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    # # model = DistilBertForSequenceClassification.from_pretrained(model_name)
    # #
    # # # Define emotions
    # # emotions = ["anger", "joy", "sadness", "fear"]
    # #
    # # def detect_emotion(text):
    # #     # Tokenize input text
    # #     inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # #
    # #     # Forward pass through the model
    # #     outputs = model(**inputs)
    # #
    # #     # Extract logits from the output
    # #     logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    # #
    # #     # Apply softmax to get probabilities
    # #     probabilities = softmax(logits, dim=1).detach().numpy()[0]
    # #
    # #     # Get the predicted emotion
    # #     predicted_emotion_index = int(torch.argmax(logits, dim=1))
    # #     predicted_emotion = emotions[predicted_emotion_index]
    # #
    # #     return predicted_emotion, probabilities
    # #
    # # # Example usage
    # # text_to_analyze =diarycontent
    # # predicted_emotion, probabilities = detect_emotion(text_to_analyze)
    # #
    # # # Display results
    # print(f"Predicted Emotion: {predicted_emotion}")
    # # print("Emotion Probabilities:")
    # # for emotion, probability in zip(emotions, probabilities):
    # #     print(f"{emotion}: {probability:.4f}")


    result = sentiment_analysis_pipeline(diarycontent)
    # print(result)


    # lid="9"
    # diarycontent=" sad "

    # import torch
    # from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    # from torch.nn.functional import softmax
    #
    # # Load pre-trained DistilBERT model and tokenizer
    # model_name = 'distilbert-base-uncased'
    # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    # model = DistilBertForSequenceClassification.from_pretrained(model_name)
    #
    # # Define emotions
    # emotions = ["anger", "joy", "sadness", "fear"]
    #
    # def detect_emotion(text):
    #     # Tokenize input text
    #     inputs = tokenizer(text, return_tensors="pt", truncation=True)
    #
    #     # Forward pass through the model
    #     outputs = model(**inputs)
    #
    #     # Extract logits from the output
    #     logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    #
    #     # Apply softmax to get probabilities
    #     probabilities = softmax(logits, dim=1).detach().numpy()[0]
    #
    #     # Get the predicted emotion
    #     predicted_emotion_index = int(torch.argmax(logits, dim=1))
    #     predicted_emotion = emotions[predicted_emotion_index]
    #
    #     return predicted_emotion, probabilities
    #
    # # Example usage
    # text_to_analyze =diarycontent
    # predicted_emotion, probabilities = detect_emotion(text_to_analyze)
    #
    # # Display results
    # print(f"Predicted Emotion: {predicted_emotion}")
    # print("Emotion Probabilities:")
    # for emotion, probability in zip(emotions, probabilities):
    #     print(f"{emotion}: {probability:.4f}")




    # from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    # import torch
    #
    # model_path = r'C:\Users\dell\PycharmProjects\Mentoring\michellejieli'
    # tokenizer_path = r'C:\Users\dell\PycharmProjects\Mentoring\michellejieli'
    #
    # model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #
    # model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    #
    # sentiment_analysis_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # # Use a pipeline as a high-level helper
    # from transformers import pipeline
    #
    # pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
    #
    #
    #
    # # diarycontent= "I love you so much"
    # #
    # # import torch
    # # from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    # # from torch.nn.functional import softmax
    # #
    # # # Load pre-trained DistilBERT model and tokenizer
    # # model_name = 'distilbert-base-uncased'
    # # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    # # model = DistilBertForSequenceClassification.from_pretrained(model_name)
    # #
    # # # Define emotions
    # # emotions = ["anger", "joy", "sadness", "fear"]
    # #
    # # def detect_emotion(text):
    # #     # Tokenize input text
    # #     inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # #
    # #     # Forward pass through the model
    # #     outputs = model(**inputs)
    # #
    # #     # Extract logits from the output
    # #     logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    # #
    # #     # Apply softmax to get probabilities
    # #     probabilities = softmax(logits, dim=1).detach().numpy()[0]
    # #
    # #     # Get the predicted emotion
    # #     predicted_emotion_index = int(torch.argmax(logits, dim=1))
    # #     predicted_emotion = emotions[predicted_emotion_index]
    # #
    # #     return predicted_emotion, probabilities
    # #
    # # # Example usage
    # # text_to_analyze =diarycontent
    # # predicted_emotion, probabilities = detect_emotion(text_to_analyze)
    # #
    # # # Display results
    # # print(f"Predicted Emotion: {predicted_emotion}")
    # # print("Emotion Probabilities:")
    # # for emotion, probability in zip(emotions, probabilities):
    # #     print(f"{emotion}: {probability:.4f}")


    # result = sentiment_analysis_pipeline("very good day")
    # print(result)
    #
    #
    # res= result[0]
    #
    try:
        lb= result['label']
    except:
        lb = result[0]['label']
    #
    #
    import  datetime
    #
    d=Diary()
    d.USER= User.objects.get(LOGIN_id=lid)
    d.content= diarycontent
    d.date= datetime.datetime.now()
    d.emotion= lb
    d.save()


    return JsonResponse({'status':'ok'})


# def user_adddiary(request):
#     lid=request.POST['lid']
#     content = request.POST['content']
#     from datetime import datetime
#     date = datetime.now()
#     obj = Diary()
#     obj.content = content
#     obj.USER = User.objects.get(LOGIN_id=lid)
#     obj.date = date
#     obj.save()
#
#
#     return JsonResponse({'status':'ok'})

def user_sendcmplnt(request):
    lid=request.POST['lid']

    complaint = request.POST['complaint']
    from datetime import datetime
    date=datetime.now()
    obj=Complaint()
    obj.complaint=complaint
    obj.USER=User.objects.get(LOGIN_id=lid)
    obj.status='pending'
    obj.reply='pending'
    obj.date=date
    obj.save()

    return JsonResponse({'status': 'ok'})

def user_viewreply(request):
    lid=request.POST['lid']
    reply=Complaint.objects.filter(USER__LOGIN_id=lid)
    l=[]
    for i in reply:
        l.append({'id':i.id,'complaint':i.complaint,'status':i.status,'date':i.date,'reply':i.reply})
    return JsonResponse({'status':'ok','data':l})


def user_viewdiary(request):
    lid=request.POST['lid']
    reply=Diary.objects.filter(USER__LOGIN_id=lid)
    l=[]
    for i in reply:
        l.append({'id':i.id,'content':i.content,'date':i.date})
    return JsonResponse({'status':'ok','data':l})




def user_sendreview(request):
    lid=request.POST['lid']
    review=request.POST['Review']
    from datetime import datetime
    date = datetime.now()
    obj = Review()
    obj.review = review
    obj.USER = User.objects.get(LOGIN_id=lid)
    obj.date = date
    obj.save()

    return JsonResponse({'status': 'ok'})

def user_changepwd(request):
    lid=request.POST['lid']
    oldpassword=request.POST['oldpassword']
    newpassword = request.POST['newpassword']
    confirmpassword = request.POST['confirmpassword']

    lobj = Login.objects.filter(id=lid, password=oldpassword)
    if lobj.exists():
        lobjj = Login.objects.get(id=lid, password=oldpassword)
        lobjj.password = newpassword
        lobjj.save()
        return JsonResponse({'status': 'ok'})
    else:


     return JsonResponse({'status':'No'})

def user_diarycam(request):
    lid=request.POST['lid']
    photo=request.FILES['file']
    # print(request.POST)
    camera=FileSystemStorage()
    from datetime import datetime
    dt=datetime.now().strftime('%Y%m%d%H%M%S')+'.jpg'
    cid=camera.save('Captures/'+str(lid)+'/'+dt,photo)

    import cv2
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights("C:\\Users\\dell\\PycharmProjects\\Mentoring\\myapp\\model.h5")

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    #for images detection from chrome
    # frame = cv2.imread(r"C:\Users\dell\Downloads\surprised.jpg")
    frame = cv2.imread("C:\\Users\\dell\\PycharmProjects\\Mentoring\\media\\Captures\\"+str(lid)+'\\'+dt)

    facecasc = cv2.CascadeClassifier("C:\\Users\\dell\\PycharmProjects\\Mentoring\\myapp\\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    result=''
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        # print(prediction)
        maxindex = int(np.argmax(prediction))
        # print(emotion_dict[maxindex])
        result= emotion_dict[maxindex]
        face=FaceEmotion()
        face.photo='/media/'+str(lid)+'/'+dt
        face.date=datetime.now().today()
        face.USER=User.objects.get(LOGIN_id=lid)
        face.emotion=result
        face.save()

    return JsonResponse({'status': 'ok'})


def user_viewtips(request):
    lid=request.POST['lid']
    res=Tips.objects.filter(USER__LOGIN_id=lid)
    l=[]
    for i in res:
        l.append({'id':i.id,'date':i.date,'tips':i.tips})
    return JsonResponse({'status': 'ok','data':l})



def user_mentoringclass(request):
    # lid = request.POST['lid']
    res = Mentoringclass.objects.all()
    l = []
    for i in res:
        l.append({'id':i.id, 'date': i.date, 'link': i.link})
    return JsonResponse({'status': 'ok', 'data': l})

def user_forgotpwd(request):
    username = request.POST['name']
    lobj = Login.objects.filter(username=username)
    if lobj.exists():
        import random
        new_pass = random.randint(0000, 9999)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("safedore3@gmail.com", "yqqlwlyqbfjtewam")  # App Password
        to = username
        subject = "Test Email"
        body = "Your new password is " + str(new_pass)
        msg = f"Subject: {subject}\n\n{body}"
        server.sendmail("s@gmail.com", to, msg)
        # Disconnect from the server
        server.quit()
        ress = Login.objects.filter(username=username).update(password=new_pass)
        return JsonResponse({'status': 'ok'})
    else:
        return HttpResponse('''<script>alert('Invalid...');window.location='/myapp/login/'</script>''')



def ConfusionMatrix(request):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix

    # Define data generators
    train_dir = 'C:\\Users\\dell\\PycharmProjects\\Mentoring\\myapp\\data\\train'
    val_dir = 'C:\\Users\\dell\\PycharmProjects\\Mentoring\\myapp\\data\\test'

    batch_size = 64

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # Load trained model weights
    model.load_weights(r'C:\\Users\\dell\\PycharmProjects\\Mentoring\\myapp\\model.h5')

    # Make predictions on the validation set
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get the actual labels
    actual_labels = validation_generator.classes

    # Compute confusion matrix
    cm = confusion_matrix(actual_labels, predicted_classes)
    # print("Confusion Matrix:")
    # print(cm)

    return render(request,"confusionMatrix.html",{"data":cm})
