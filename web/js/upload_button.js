console.log("FilmScanEmulator JS loaded");

if (window.comfyAPI?.app && window.comfyAPI?.api) {
    const { app } = window.comfyAPI.app;
    const { api } = window.comfyAPI.api;

    app.registerExtension({
        name: "FilmScanEmulator.UploadButton",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== "AET_ImageLoader" && nodeData.name !== "AET_ReferenceLoader") return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const widget = this.widgets?.find(w => w.name === "image");
                if (!widget) return r;

                // Додаємо стандартний віджет-кнопку, який ComfyUI вміє малювати
                const btnWidget = this.addWidget("button", "📁 Upload", null, async () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "image/*";
                    input.onchange = async () => {
                        const file = input.files[0];
                        if (!file) return;
                        const form = new FormData();
                        form.append("image", file);
                        const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
                        const data = await resp.json();
                        widget.value = data.name;
                        if (widget.callback) widget.callback(data.name);
                    };
                    input.click();
                });
                btnWidget.serialize = false; // не зберігати в workflow

                // Прев'ю — окремий DOM-елемент, але вже без кнопки
                const preview = document.createElement("img");
                preview.style.cssText = "max-width:100%;max-height:180px;object-fit:contain;display:none;border-radius:4px;";
                this.addDOMWidget("preview", "img", preview, {
                    serialize: false,
                    setValue: (value) => {
                        if (value) {
                            preview.src = `/view?filename=${encodeURIComponent(value)}&type=input&subfolder=`;
                            preview.style.display = "block";
                        } else {
                            preview.style.display = "none";
                        }
                    },
                });

                // Оновлення прев'ю при зміні widget.value
                const originalCallback = widget.callback;
                widget.callback = function (value) {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    if (value) {
                        preview.src = `/view?filename=${encodeURIComponent(value)}&type=input&subfolder=`;
                        preview.style.display = "block";
                    } else {
                        preview.style.display = "none";
                    }
                };

                if (widget.value) {
                    preview.src = `/view?filename=${encodeURIComponent(widget.value)}&type=input&subfolder=`;
                    preview.style.display = "block";
                }

                return r;
            };
        }
    });
}