// Bug Management System — shared report interactions.
// The only behaviour these static reports need: click a table header to sort
// it. Numeric columns carry their raw value in data-sort; everything else
// sorts on its own text.
(function () {
  function cellValue(td) {
    if (td.dataset.sort !== undefined) return parseFloat(td.dataset.sort);
    var text = td.textContent.trim().replace(/[,%]/g, '');
    var num = parseFloat(text);
    return isNaN(num) ? td.textContent.trim().toLowerCase() : num;
  }

  document.querySelectorAll('table[data-sortable]').forEach(function (table) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var headers = Array.prototype.slice.call(table.tHead.rows[0].cells);

    headers.forEach(function (th, colIndex) {
      if (th.dataset.nosort !== undefined) return;
      th.addEventListener('click', function () {
        var ascending = !th.classList.contains('sorted') || th.classList.contains('sorted-asc');
        headers.forEach(function (h) { h.classList.remove('sorted', 'sorted-asc'); });
        th.classList.add('sorted');
        if (ascending) th.classList.add('sorted-asc');

        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) {
          var av = cellValue(a.cells[colIndex]);
          var bv = cellValue(b.cells[colIndex]);
          if (av < bv) return ascending ? -1 : 1;
          if (av > bv) return ascending ? 1 : -1;
          return 0;
        });
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });
  });
})();
