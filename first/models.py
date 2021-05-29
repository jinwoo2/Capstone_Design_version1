# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import UserManager
from django.db import models



class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Building(models.Model):
    buildingcode = models.CharField(db_column='BuildingCode', primary_key=True, max_length=10)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'building'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class FirstSubjectkeyword(models.Model):
    keyword_id = models.IntegerField()
    keyword = models.TextField()

    class Meta:
        managed = False
        db_table = 'first_subjectkeyword'


class FirstUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    userid = models.TextField()
    username = models.TextField()
    user_number = models.TextField()

    class Meta:
        managed = False
        db_table = 'first_user'


class FirstUserkeyword(models.Model):
    user_id = models.IntegerField()
    keyword_id = models.IntegerField()
    keyword = models.TextField()
    flag = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'first_userkeyword'


class Kiosk(models.Model):
    kioskcode = models.CharField(db_column='KioskCode', primary_key=True, max_length=10)  # Field name made lowercase.
    buildingcode = models.ForeignKey(Building, on_delete=models.CASCADE, db_column='BuildingCode')  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'kiosk'


class Kioskaccess(models.Model):
    accesscode = models.CharField(db_column='AccessCode', primary_key=True, max_length=10)  # Field name made lowercase.
    kioskcode = models.ForeignKey(Kiosk, on_delete=models.CASCADE, db_column='KioskCode')  # Field name made lowercase.
    timedata = models.DateTimeField(db_column='TimeData')  # Field name made lowercase.
    id = models.ForeignKey('Person', on_delete=models.CASCADE, db_column='ID')  # Field name made lowercase.
    staticskey = models.ManyToManyField('Statics', db_column='staticskey')

    class Meta:
        managed = False
        db_table = 'kioskaccess'


class Kstate(models.Model):
    kioskcode = models.OneToOneField(Kiosk, on_delete=models.CASCADE, db_column='KioskCode', primary_key=True)  # Field name made lowercase.
    inspectorcode = models.IntegerField(db_column='InspectorCode')  # Field name made lowercase.
    inspectdate = models.DateField(db_column='InspectDate', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'kstate'
        unique_together = (('kioskcode', 'inspectorcode'),)


class Person(models.Model):
    id = models.IntegerField(db_column='ID', primary_key=True)  # Field name made lowercase.
    name = models.CharField(db_column='Name', max_length=10)  # Field name made lowercase.
    address = models.CharField(db_column='Address', max_length=30, blank=True, null=True)  # Field name made lowercase.
    phone = models.CharField(db_column='Phone', max_length=20, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'person'


class State(models.Model):
    stateid = models.CharField(db_column='StateID', primary_key=True, max_length=10)  # Field name made lowercase.
    id = models.OneToOneField(Person, on_delete=models.CASCADE, db_column='ID')  # Field name made lowercase.
    datetimedata = models.DateTimeField(db_column='DateTimedata')
    infoagree = models.IntegerField(db_column='InfoAgree')  # Field name made lowercase.
    temperature = models.FloatField(db_column='Temperature')  # Field name made lowercase.
    breathingsymptoms = models.IntegerField(db_column='BreathingSymptoms')  # Field name made lowercase.
    abnormaltemperature = models.IntegerField(db_column='AbnormalTemperature', blank=True, null=True)  # Field name made lowercase.
    confirm = models.IntegerField(db_column='Confirm')
    suspect = models.IntegerField(db_column='Suspect')


    class Meta:
        managed = False
        db_table = 'state'
        unique_together = (('stateid', 'id'),)


class Statics(models.Model):
    staticskey = models.IntegerField(db_column='StaticsKey', primary_key=True)  # Field name made lowercase.
    day = models.DateField(db_column='Day')  # Field name made lowercase.
    e87enter = models.IntegerField(db_column='E87Enter', blank=True, null=True)  # Field name made lowercase.
    n14enter = models.IntegerField(db_column='N14Enter', blank=True, null=True)  # Field name made lowercase.
    s20enter = models.IntegerField(db_column='S20Enter', blank=True, null=True)  # Field name made lowercase.
    s41enter = models.IntegerField(db_column='S41Enter', blank=True, null=True)  # Field name made lowercase.
    e87confirm = models.IntegerField(db_column='E87Confirm', blank=True, null=True)  # Field name made lowercase.
    s41confirm = models.IntegerField(db_column='S41Confirm', blank=True, null=True)  # Field name made lowercase.
    n14confirm = models.IntegerField(db_column='N14Confirm', blank=True, null=True)  # Field name made lowercase.
    s20confirm = models.IntegerField(db_column='S20Confirm', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'statics'



class User(AbstractBaseUser):
    userid = models.TextField(max_length=20) # 계정
    username = models.TextField(max_length=20)
    user_number = models.TextField(max_length=10)
    objects = UserManager()

    USERNAME_FIELD = 'userid'
    REQUIRED_FIELDS = ['user_number']

    def __str__(self):
        return self.username


class SubjectKeyword(models.Model):
    keyword_id = models.IntegerField()
    keyword = models.TextField(max_length=10)


class UserKeyword(models.Model):
    user_id = models.IntegerField()
    keyword_id = models.IntegerField()
    keyword = models.TextField(max_length=10)
    flag = models.IntegerField()