# Generated by Django 3.2.24 on 2024-02-22 06:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='diary',
            name='emotion',
            field=models.CharField(default='', max_length=100),
        ),
    ]
