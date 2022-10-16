# Generated by Django 3.2.5 on 2022-10-14 15:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BookMetadata',
            fields=[
                ('comic_no', models.CharField(max_length=6, primary_key=True, serialize=False)),
                ('book_title', models.CharField(blank=True, max_length=50)),
                ('volume', models.CharField(blank=True, max_length=4, null=True)),
                ('issue', models.CharField(blank=True, max_length=4, null=True)),
                ('year', models.CharField(blank=True, max_length=5, null=True)),
                ('month', models.CharField(blank=True, max_length=20, null=True)),
                ('genre', models.CharField(blank=True, max_length=50, null=True)),
                ('weblink', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Drink',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('description', models.CharField(max_length=500)),
            ],
        ),
    ]
