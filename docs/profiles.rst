.. profiles.rst
.. py:module:: dit.profiles

********************
Information Profiles
********************

There are several ways to decompose the information contained in a joint distribution. Here, we will demonstrate their behavior using four examples drawn from :cite:`Allen2014`:

.. ipython::

   In [1]: from dit.profiles import *

   In [2]: ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)

   In [3]: ex2 = Distribution(['000', '111'], [1/2]*2)

   In [4]: ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)

   In [5]: ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)

Complexity Profile
==================

The complexity profile is simply the sum of each "layer" of the I-diagram :cite:`Baryam2004`.

.. image:: ../images/profiles/complexity_profile_example_1.png
   :alt: The complexity profile for example 1
   :align: center

.. image:: ../images/profiles/complexity_profile_example_2.png
   :alt: The complexity profile for example 2
   :align: center

.. image:: ../images/profiles/complexity_profile_example_3.png
   :alt: The complexity profile for example 3
   :align: center

.. image:: ../images/profiles/complexity_profile_example_4.png
   :alt: The complexity profile for example 4
   :align: center

Marginal Utility of Information
===============================

:cite:`Allen2014`

.. image:: ../images/profiles/mui_profile_example_1.png
   :alt: The MUI profile for example 1
   :align: center

.. image:: ../images/profiles/mui_profile_example_2.png
   :alt: The MUI profile for example 2
   :align: center

.. image:: ../images/profiles/mui_profile_example_3.png
   :alt: The MUI profile for example 3
   :align: center

.. image:: ../images/profiles/mui_profile_example_4.png
   :alt: The MUI profile for example 4
   :align: center

Schneidman Profile
==================

Also known as the *connected information* or *network informations*, the Schneidman profile exposes how much information is learned about the distribution when considering :math:`k`-way dependencies :cite:`Schneidman2003`.

.. image:: ../images/profiles/schneidman_profile_example_1.png
   :alt: The Schneidman profile for example 1
   :align: center

.. image:: ../images/profiles/schneidman_profile_example_2.png
   :alt: The Schneidman profile for example 2
   :align: center

.. image:: ../images/profiles/schneidman_profile_example_3.png
   :alt: The Schneidman profile for example 3
   :align: center

.. image:: ../images/profiles/schneidman_profile_example_4.png
   :alt: The Schneidman profile for example 4
   :align: center
