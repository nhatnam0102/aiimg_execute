import sys
import  os
import logging
logging.basicConfig(stream=sys.stderr)
activate_this = '/home/upc/WorkSpaces/nam/aiimg/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))
sys.path.insert(0, "/var/www/html/aiimg_execute")
from aiimg_execute import app as application
application.secret_key = 'your_secret_key'
