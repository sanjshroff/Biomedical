import ssl
# Fix: disabling SSL, because the nltk package was not downloading on MAC OS
def ssl_disable():
    try:
       _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context