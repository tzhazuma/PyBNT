
#Thanks to the so much third-lib in python
#Thanks to the Dubai Tongyi and Deepseek AI !!!
#You are my robot teacher in using some third-lib!!!

from GUI_main import *

def main()->int:
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    ui.run();
    Dialog.show()
    sys.exit(app.exec())
    return 0;

if(__name__=="__main__"):
    exit(main());