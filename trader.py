import getpass
class Trader:
    API_NAME = None
    def __init__(self):
        self.login()

    def login(self):
        """
        Login to trading API
        :return:
        """

        user = input("Login for " + self.API_NAME + ":")
        # Don't store pass in variable
        pw = getpass.getpass("Password:")
        pass

    def isDown(self):
        """
        Check if trading API is down
        :return: boolean
        """
        return False

    def execute(self):
        if self.isDown():
            return False
        pass