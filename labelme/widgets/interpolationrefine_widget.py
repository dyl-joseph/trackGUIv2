from qtpy import QtWidgets


class IterpolationRefineWidget(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(IterpolationRefineWidget, self).__init__(parent)

        self.button = QtWidgets.QPushButton("Edit")

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setStyleSheet(
            "border: 1px solid black; border-radius: 1px; text-align: center;"
        )
        self.statusBar.showMessage("Name: None | ID: None")

        self.checkBox = QtWidgets.QCheckBox()

        navigationLayout = QtWidgets.QHBoxLayout()
        navigationLayout.addWidget(self.statusBar)
        navigationLayout.addWidget(self.checkBox)
        navigationLayout.addWidget(self.button)

        self.setLayout(navigationLayout)
