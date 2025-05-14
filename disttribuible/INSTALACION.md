# Guía de Instalación y Uso de Fast-Agent-MCP

Este documento explica cómo instalar y usar el paquete `fast_agent_mcp` mientras esperas la aprobación de un PR o si necesitas usar una versión específica del paquete en otro dispositivo.

## Opciones de Instalación

### 1. Usar los Archivos Distribuibles (Recomendado)

Los archivos distribuibles (wheel y tarball) permiten instalar el paquete fácilmente en cualquier dispositivo:

```bash
# Usando el archivo wheel
pip install path/to/fast_agent_mcp-0.2.24-py3-none-any.whl

# O usando el archivo tar.gz
pip install path/to/fast_agent_mcp-0.2.24.tar.gz
```

### 2. Instalación con Poetry

Si utilizas Poetry para la gestión de dependencias:

```bash
# Usando el archivo wheel
poetry add path/to/fast_agent_mcp-0.2.24-py3-none-any.whl

# O directamente desde un repositorio Git
poetry add git+https://github.com/tu-usuario/fast-agent.git@tu-rama
```

### 3. Desde un Repositorio PyPI Privado

Si has subido el paquete a un repositorio PyPI privado:

```bash
# Agregar el repositorio a Poetry (si es necesario)
poetry source add mi-pypi https://tu-pypi-privado.com/simple/

# Instalar el paquete
poetry add --source mi-pypi fast-agent-mcp==0.2.24
```

## Crear los Archivos Distribuibles

Si necesitas generar los archivos distribuibles desde el código fuente:

```bash
# Instalar build si no está instalado
pip install build

# Generar los archivos distribuibles
python -m build

# Los archivos se crearán en el directorio "dist"
```

## Ejemplos de Uso

### Ejemplo Básico

```python
from mcp_agent import Agent
from mcp_agent.config import AgentConfig

# Configurar el agente
config = AgentConfig()
agent = Agent(config)

# Ejecutar una consulta
response = agent.prompt("Analiza los datos de ventas y resúmelos brevemente")
print(response)
```

### Uso con Azure OpenAI

Como se muestra en el directorio `examples/azure-openai`, puedes configurar el agente para usar Azure OpenAI:

```python
from mcp_agent import Agent
from mcp_agent.config import AgentConfig

# Cargar configuración desde un archivo YAML
config = AgentConfig.from_file("fastagent.config.yaml")
agent = Agent(config)

# Ejecutar una consulta
response = agent.prompt("Genera una breve historia sobre un viaje espacial")
print(response)
```

### Análisis de Datos

Para escenarios de análisis de datos, como en el directorio `examples/data-analysis`:

```python
from mcp_agent import Agent
from mcp_agent.config import AgentConfig

# Configurar el agente
config = AgentConfig.from_file("fastagent.config.yaml")
agent = Agent(config)

# Ejecutar consulta de análisis
response = agent.prompt("Analiza el archivo CSV de attrition y proporciona insights clave")
print(response)
```

## Workflows Avanzados

Para casos de uso más avanzados, puedes utilizar los workflows como se muestra en el directorio `examples/workflows`:

```python
from mcp_agent import Agent, Workflow
from mcp_agent.config import AgentConfig

# Configurar el workflow
config = AgentConfig.from_file("fastagent.config.yaml")
workflow = Workflow(config)

# Definir pasos del workflow
workflow.add_step("investigación", "Investiga sobre energías renovables")
workflow.add_step("análisis", "Analiza los datos recolectados en el paso anterior")
workflow.add_step("conclusión", "Genera una conclusión basada en los análisis")

# Ejecutar el workflow
results = workflow.run()
for step, result in results.items():
    print(f"== {step} ==")
    print(result)
```

## Solución de Problemas

Si encuentras problemas al instalar o usar el paquete:

1. Verifica que estás usando Python 3.8 o superior
2. Asegúrate de que todas las dependencias están instaladas correctamente
3. Revisa los logs para información detallada sobre los errores

## Recursos Adicionales

- Consulta los ejemplos en el directorio `examples/` para ver más casos de uso
- Revisa los tests en `tests/` para entender el funcionamiento interno del paquete

---

Para más información, consulta la documentación completa del proyecto o abre un issue en el repositorio.
