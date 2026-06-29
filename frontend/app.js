const room = document.getElementById("room-container");

function addFurniture(type) {
  const img = document.createElement("img");
  img.src = `assets/furniture/${type}.png`;
  img.className = "furniture";

  img.style.left = "50px";
  img.style.top = "50px";

  makeDraggable(img);
  room.appendChild(img);
}

function makeDraggable(element) {
  let offsetX, offsetY, isDragging = false;

  element.addEventListener("mousedown", (e) => {
    isDragging = true;
    offsetX = e.offsetX;
    offsetY = e.offsetY;
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    const rect = room.getBoundingClientRect();
    element.style.left = e.clientX - rect.left - offsetX + "px";
    element.style.top = e.clientY - rect.top - offsetY + "px";
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
  });
}
