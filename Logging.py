class Log:

  def __init__(self, logging, name):
    self.name = name
    self.logging = logging

  def message(self, message, **kwargs):
    conditioned =  kwargs.get('conditioned', False)
    forced = kwargs.get('forced', False)

    if self.logging and not conditioned:
      print self.name + ": " + str(message)

    if forced:
      print self.name + ": " + str(message)
