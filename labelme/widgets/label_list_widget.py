from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QStyle


# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.HighlightedText),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        thefuckyourshitup_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - thefuckyourshitup_constant
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        thefuckyourshitup_constant = 4
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - thefuckyourshitup_constant),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.setShape(shape)

        self.setCheckable(True)
        self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class StandardItemModel(QtGui.QStandardItemModel):
    pass


class LabelListWidget(QtWidgets.QListView):
    itemDoubleClicked = QtCore.Signal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.Signal(list, list)
    itemDropped = QtCore.Signal()

    def __init__(self):
        super(LabelListWidget, self).__init__()
        self._selectedItems = []
        self._shape_to_item = {}

        self.setWindowFlags(Qt.Window)
        self.setModel(StandardItemModel())
        self.model().setItemPrototype(LabelListWidgetItem())
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)

    def __len__(self):
        return self.model().rowCount()

    def __getitem__(self, i):
        return self.model().item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def itemChanged(self):
        return self.model().itemChanged

    def dropEvent(self, event):
        before = tuple(id(item.shape()) for item in self)
        super().dropEvent(event)
        if event.isAccepted():
            self._scheduleItemDropped(before)

    def _scheduleItemDropped(self, before):
        # With InternalMove, Qt inserts the target clone during dropEvent but
        # removes the source row only after dropEvent returns. Inspect the final
        # model on the next event-loop turn.
        QtCore.QTimer.singleShot(0, lambda: self._emitItemDroppedIfChanged(before))

    def _emitItemDroppedIfChanged(self, before):
        after = tuple(id(item.shape()) for item in self)
        if after != before:
            self.itemDropped.emit()

    def itemSelectionChangedEvent(self, selected, deselected):
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]
        deselected = [self.model().itemFromIndex(i) for i in deselected.indexes()]
        self.itemSelectionChanged.emit(selected, deselected)

    def itemDoubleClickedEvent(self, index):
        self.itemDoubleClicked.emit(self.model().itemFromIndex(index))

    def selectedItems(self):
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]

    def scrollToItem(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self.model().setItem(self.model().rowCount(), 0, item)
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))
        shape = item.shape()
        if shape is not None:
            self._shape_to_item[id(shape)] = item

    def removeItem(self, item):
        shape = item.shape()
        if shape is not None:
            self._shape_to_item.pop(id(shape), None)
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def selectItem(self, item):
        index = self.model().indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def findItemByShape(self, shape):
        item = self._shape_to_item.get(id(shape))
        if item is not None:
            try:
                index = self.model().indexFromItem(item)
                if (
                    index.isValid()
                    and self.model().itemFromIndex(index) is item
                    and item.shape() is shape
                ):
                    return item
            except RuntimeError:
                # Some Qt bindings delete the wrapped C++ item as soon as the
                # source row of an internal move is removed.
                pass
            # Qt implements internal moves by cloning the item and detaching the
            # original, so cached items must never be trusted without checking
            # that they still belong to this model.
            self._shape_to_item.pop(id(shape), None)
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() is shape:
                self._shape_to_item[id(shape)] = item
                return item
        raise ValueError("cannot find shape: {}".format(shape))

    def clear(self):
        self._shape_to_item.clear()
        self.model().clear()
