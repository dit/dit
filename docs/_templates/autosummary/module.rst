{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
Classes
-------

.. autosummary::
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
Functions
---------

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
Exceptions
----------

.. autosummary::
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
Data
----

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes or functions or exceptions %}
Reference
---------
{% endif %}

{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:
{% endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}

{% for item in exceptions %}
.. autoexception:: {{ item }}
{% endfor %}

{% for item in attributes %}
.. autodata:: {{ item }}
{% endfor %}
