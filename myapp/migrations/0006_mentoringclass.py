# Generated by Django 3.2.24 on 2024-04-22 03:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0005_auto_20240411_1041'),
    ]

    operations = [
        migrations.CreateModel(
            name='Mentoringclass',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('link', models.CharField(max_length=500)),
            ],
        ),
    ]
