from SimpleXMLRPCServer import SimpleXMLRPCServer

class ImseXMLRPCServer(SimpleXMLRPCServer):
	def __init__(self, addr):
		SimpleXMLRPCServer.__init__(self, addr)
		f = open("/ldata/Imse/port.txt", "w")
		f.write(str(self.server_address[1]))
		f.close()
