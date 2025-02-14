# 📌 Guía de Comandos Git y Autenticación en GitHub

## 🚀 1. Configuración de Autenticación
### 🔑 *Autenticación con Token de Acceso Personal (PAT)*
1.⁠ ⁠Ir a [GitHub → Settings](https://github.com/settings/profile).
2.⁠ ⁠En el menú de la izquierda, ir a *Developer settings* → *Personal access tokens*.
3.⁠ ⁠Hacer clic en *"Generate new token (classic)"*.
4.⁠ ⁠Seleccionar permisos:
   - ⁠ repo ⁠ (para acceder a repositorios).
   - ⁠ workflow ⁠ (para GitHub Actions, opcional).
5.⁠ ⁠Generar y copiar el token.
6.⁠ ⁠Usar este formato para clonar el repositorio:
   ⁠ bash
   git clone https://TOKEN@github.com/usuario/repositorio.git
    ⁠
   
### 🔐 *Autenticación con SSH*
1.⁠ ⁠*Generar una clave SSH*
   ⁠ bash
   ssh-keygen -t rsa -b 4096 -C "tu-email@example.com"
    ⁠
   Presiona *Enter* para aceptar la ubicación predeterminada (~/.ssh/id_rsa).
2.⁠ ⁠*Iniciar el agente SSH*
   ⁠ bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
    ⁠
3.⁠ ⁠*Añadir la clave pública a GitHub*
   ⁠ bash
   cat ~/.ssh/id_rsa.pub
    ⁠
   Copia la clave y agrégala en [GitHub → SSH keys](https://github.com/settings/keys).
4.⁠ ⁠*Probar la conexión*
   ⁠ bash
   ssh -T git@github.com
    ⁠
   Si ves ⁠ Hi usuario! You've successfully authenticated. ⁠, está configurado correctamente.

---

## 🔄 2. Flujo de Trabajo Básico en GitHub

### 🔃 *Actualizar el Repositorio Local*
Antes de realizar cambios, siempre actualiza tu repositorio local:
⁠ bash
git pull origin main
 ⁠
💡 *Importancia de ⁠ git pull ⁠*: Evita conflictos entre versiones y mantiene tu código actualizado con el trabajo de otros colaboradores.

### ✍️ *Hacer Cambios y Subirlos a GitHub*
1.⁠ ⁠Agregar archivos al área de staging:
   ⁠ bash
   git add .
    ⁠
2.⁠ ⁠Hacer un commit con un mensaje descriptivo:
   ⁠ bash
   git commit -m "Descripción de los cambios"
    ⁠
3.⁠ ⁠Subir los cambios a GitHub:
   ⁠ bash
   git push origin main
    ⁠

---

## 🔄 3. Trabajo con Ramas (Opcional)
Si trabajas en equipo, es recomendable usar ramas para evitar conflictos.

1.⁠ ⁠Crear una nueva rama:
   ⁠ bash
   git checkout -b nombre-rama
    ⁠
2.⁠ ⁠Subir cambios en la nueva rama:
   ⁠ bash
   git push origin nombre-rama
    ⁠
3.⁠ ⁠Crear un *Pull Request* en GitHub y fusionar con ⁠ main ⁠.

---

🎉 ¡Ahora estás listo para trabajar con Git y GitHub de manera eficiente!