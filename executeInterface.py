class ExecuteInterface:
    """
    Interface for backtest and liveExecute
    """

    def __init__(self, cash, data):
        raise NotImplementedError("Execute interface constructor")

    def buy(self, amount):
        raise NotImplementedError("Execution interface buy")

    def sell(self, amount):
        raise NotImplementedError("Execution interface sell")

    def doNothing(self):
        raise NotImplementedError("Execution interface doNothing")

    def value(self):
        raise NotImplementedError("Execution interface value")