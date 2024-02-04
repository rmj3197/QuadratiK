{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

.. rubric:: Methods

.. autosummary::
    {% for item in members %}
    {% if item in ['__call__'] %}
        {{ objname }}.{{ item }}
    {% endif %}
    {% endfor %}
    {% for item in methods %}
    {% if item != '__init__' %}
        {{ objname }}.{{ item }}
    {% endif %}
    {% endfor %}

----

{% for item in methods %}
{% if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

.. raw:: html

     <div style='clear:both'></div>