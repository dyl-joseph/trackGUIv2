from qtpy import QtWidgets


class DeletionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(DeletionDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Modification Options")

        self.start_frame_cell = QtWidgets.QLineEdit()
        self.end_frame_cell = QtWidgets.QLineEdit()
        self.ID_cell = QtWidgets.QLineEdit()
        self.label_cell = QtWidgets.QLineEdit()
        self.new_ID_cell = QtWidgets.QLineEdit()
        self.new_label_cell = QtWidgets.QLineEdit()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Remove Box", "Swap Label", "Swap ID"])
        self.mode_combo.setCurrentText("Swap ID")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Start Frame:", self.start_frame_cell)
        layout.addRow("End Frame:", self.end_frame_cell)
        layout.addRow("Object ID:", self.ID_cell)
        layout.addRow("Object Label:", self.label_cell)
        layout.addRow("New ID:", self.new_ID_cell)
        layout.addRow("New Label:", self.new_label_cell)
        layout.addRow("Mode:", self.mode_combo)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    @property
    def mode(self):
        return self.mode_combo.currentText()
