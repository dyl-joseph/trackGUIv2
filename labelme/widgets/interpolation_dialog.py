from dataclasses import dataclass

from qtpy import QtWidgets


@dataclass(frozen=True)
class InterpolationOptions:
    start_frame: int
    end_frame: int
    interval: int
    track_id: str
    label: str


class InterpolationDialog(QtWidgets.QDialog):
    def __init__(self, min_val, max_val, parent=None):
        super(InterpolationDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Interpolation Options")

        minimum = max(1, int(min_val))
        maximum = max(minimum, int(max_val))
        self.start_frame_cell = QtWidgets.QSpinBox()
        self.start_frame_cell.setRange(minimum, maximum)
        self.start_frame_cell.setValue(minimum)
        self.end_frame_cell = QtWidgets.QSpinBox()
        self.end_frame_cell.setRange(minimum, maximum)
        self.end_frame_cell.setValue(maximum)
        self.interval_cell = QtWidgets.QSpinBox()
        self.interval_cell.setRange(1, max(1, maximum - minimum))
        self.interval_cell.setValue(1)
        self.ID_cell = QtWidgets.QLineEdit()
        self.label_cell = QtWidgets.QLineEdit()

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Start Frame:"))
        row1.addStretch()
        row1.addWidget(self.start_frame_cell)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("End Frame:"))
        row2.addStretch()
        row2.addWidget(self.end_frame_cell)

        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Interval/FPS:"))
        row3.addStretch()
        row3.addWidget(self.interval_cell)

        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(QtWidgets.QLabel("Object ID:"))
        row4.addStretch()
        row4.addWidget(self.ID_cell)

        row5 = QtWidgets.QHBoxLayout()
        row5.addWidget(QtWidgets.QLabel("Object Label"))
        row5.addStretch()
        row5.addWidget(self.label_cell)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addLayout(row4)
        layout.addLayout(row5)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def options(self):
        track_id = self.ID_cell.text().strip()
        label = self.label_cell.text().strip()
        if not track_id or not label:
            raise ValueError("Object label and ID are required.")
        return InterpolationOptions(
            start_frame=self.start_frame_cell.value(),
            end_frame=self.end_frame_cell.value(),
            interval=self.interval_cell.value(),
            track_id=track_id,
            label=label,
        )
