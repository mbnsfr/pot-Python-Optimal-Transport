run this command on main directory

pip install -r requirements.txt

if you want to run python of the project virtual then run this command on main folder

virtualenv venv -p python<your python version>
source venv/bin/activate

now in <potExample> folder for run project run this command

if you installed and activated virtualenv :

python manage.py runserver

else :

python<your python version> manage.py runserver
