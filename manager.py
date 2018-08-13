
class Manager:
    def __init__(self, info, agent, trader):
        self.__info = info
        self.__agent = agent
        self.__trader = trader

    def run(self):
        """
        Liaison between info, agent, and trader modules
        :return: Null
        """
        stop_flag = False

        while not stop_flag:
            info = self.__info.new()
            decision = self.__agent.update(info)
            if not decision == None:
                success = self.__trader.execute(decision)
            else:
                success = True
            stop_flag = self.checkStop()

    def checkStop(self):
        """
        Checks for interrupt
        :return: boolean indicating interrupt
        """
        return False